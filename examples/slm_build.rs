//! ScribeSLM's flagship language model, built from scratch on standards:
//! people who work on the language (lexicographers: Wiktionary and WordNet
//! examples; annotators: the EWT treebank) and learners (Tatoeba; Simple
//! English Wikipedia, written for learners). No pretrained weights, no
//! published-author or web text, no definitions.
//!
//!   cargo run --release --features metal --example slm_build [steps] [--sample N] [--vocab 30000]
//!       [--joint 1.0] [--out models/curated-scratch] [--resume]
//!
//! - A WordPiece vocabulary learned from this corpus (src/wordpiece.rs).
//! - Same shape as BERT-mini (4 layers × 256, 4 heads, 1,024 feed-forward), so it
//!   runs wherever BERT-mini does; loads with bert::Bert.
//! - Pre-LayerNorm layout (trains reliably from scratch; --post-ln for BERT's).
//! - Whole-word masking, many passes, and (--joint) the dictionary objective:
//!   a hidden SemCor word's slot should accept its labelled meaning's synonyms.
//! - --sample N: N sentences per source (smoke tests). --every N: held-out
//!   check every N steps (default 2,000).
//! - --quick: a ~3-minute recipe check (1,500 steps on 5,000 sentences per
//!   source, 8,000-piece vocabulary, a reading every 250 steps). For "does this
//!   recipe learn, and how fast?", not for scores.
//! - --focus-swords ADJ|NOUN|VERB|ADV [--cap 300000]: a topic slice for quick
//!   recipe tests. Focus words are the SWORDS test's target words of that part
//!   of speech (never its answer lists) plus their WordNet synonyms and similar
//!   words; training keeps every standards sentence holding a focus word (up to
//!   --cap) plus the --sample background. Score with swords_eval POS=<same>.
//! Keeps the step with the lowest held-out loss: 2,000 Tatoeba sentences and
//! 1,000 SemCor words never trained on.
use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use scribe::bert::Bert;
use scribe::slm_train::{dictionary_examples, dictionary_loss, mask_whole_words, masked_loss, DictExample, Encoded, Rng};
use scribe::usage::corpus;
use scribe::{conllu, text, wordpiece, Lexicon, Pos};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Default sentences per step (--batch to change).
const BATCH: usize = 128;
const HELD_OUT: usize = 2000;
const WARMUP: usize = 2000;
const LR: f64 = 5e-4;
const MAX_LEN: usize = 128;
const SEMCOR: &str = "data/semcor/wsdeval/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor";

/// SWORDS test target words of one part of speech, with their WordNet
/// synonyms (every meaning) and, for adjectives, similar words.
fn focus_words(lex: &Lexicon, pos: &str) -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    let d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string("data/swords/swords-v1.1_test.json")?)?;
    let p = match pos { "NOUN" => Pos::Noun, "VERB" => Pos::Verb, "ADJ" => Pos::Adj, "ADV" => Pos::Adv, _ => return Err("POS must be ADJ, NOUN, VERB or ADV".into()) };
    let mut out = HashSet::new();
    for t in d["targets"].as_object().ok_or("no targets")?.values() {
        if t["pos"].as_str() != Some(pos) { continue }
        let word = t["target"].as_str().unwrap_or("").to_lowercase();
        out.insert(word.clone());
        let lemma = lex.lemmatize(&word, Some(p)).into_iter().next().unwrap_or(word);
        for &s in lex.senses(&lemma, p) {
            let syn = &lex.synsets[s as usize];
            let similar = syn.similar.iter().flat_map(|&x| lex.synsets[x as usize].members.iter());
            out.extend(syn.members.iter().chain(similar).filter(|m| !m.contains(' ')).map(|m| m.to_lowercase()));
        }
        out.insert(lemma);
    }
    Ok(out)
}

/// Every `len / n`-th item (a spread sample), or all of them.
fn sample<T: Clone>(v: Vec<T>, n: Option<usize>) -> Vec<T> {
    match n {
        Some(n) if n < v.len() => { let step = v.len() / n; v.into_iter().step_by(step).take(n).collect() }
        _ => v,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(if quick { 1_500 } else { 200_000 });
    let value = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
    let n: Option<usize> = value("--sample").and_then(|v| v.parse().ok()).or(quick.then_some(5_000));
    let vocab_size: usize = value("--vocab").and_then(|v| v.parse().ok()).unwrap_or(if quick { 8_000 } else { 30_000 });
    let batch_size: usize = value("--batch").and_then(|v| v.parse().ok()).unwrap_or(BATCH);
    let every: usize = value("--every").and_then(|v| v.parse().ok()).unwrap_or(if quick { 250 } else { 2_000 });
    let joint: f32 = value("--joint").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let out = PathBuf::from(value("--out").unwrap_or_else(|| "models/curated-scratch".into()));
    let resume = args.iter().any(|a| a == "--resume");
    let focus_pos = value("--focus-swords");
    let cap: usize = value("--cap").and_then(|v| v.parse().ok()).unwrap_or(300_000);
    #[cfg(feature = "metal")]
    let dev = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let dev = Device::Cpu;

    // Standards only, each source sampled the same way.
    let lex = Lexicon::load(Path::new("models/lexicon.bin"))?;
    let lines = |p: &str| -> Result<Vec<Vec<String>>, std::io::Error> { Ok(std::fs::read_to_string(p)?.lines().map(text::tokenize).filter(|t| !t.is_empty()).collect()) };
    let mut tatoeba = lines("data/corpus/tatoeba.txt")?;
    let held: Vec<Vec<String>> = tatoeba.drain(..HELD_OUT).collect();
    let full: Vec<(&str, Vec<Vec<String>>)> = vec![
        ("wiktionary examples", lines("data/wiktionary/examples.txt")?),
        ("EWT treebank", conllu::parse(&std::fs::read_to_string("data/en_ewt-ud-train.conllu")?).iter().map(|s| s.iter().map(|t| t.form.clone()).collect()).collect()),
        ("WordNet examples", corpus::wordnet_examples(&lex).into_iter().filter(|e| !corpus::is_held_out(e.synset, &e.sentence, 10)).map(|e| text::tokenize(&e.sentence)).collect()),
        ("Tatoeba", tatoeba),
        ("Simple English Wikipedia", lines("data/corpus/simplewiki.txt")?),
    ];
    // Topic slice: sentences holding a focus word (capped), plus the background sample.
    let focus = match &focus_pos { Some(p) => Some(focus_words(&lex, p)?), None => None };
    let mut focused: Vec<Vec<String>> = Vec::new();
    let mut sources: Vec<(&str, Vec<Vec<String>>)> = Vec::new();
    for (name, sents) in full {
        match &focus {
            Some(f) => {
                let (hit, rest): (Vec<_>, Vec<_>) = sents.into_iter().partition(|s| s.iter().any(|w| f.contains(&w.to_lowercase())));
                eprintln!("  {name}: {} sentences hold a focus word", hit.len());
                focused.extend(hit);
                sources.push((name, sample(rest, n)));
            }
            None => sources.push((name, sample(sents, n))),
        }
    }
    if let (Some(f), Some(p)) = (&focus, &focus_pos) {
        let mut rng = Rng(17);
        for i in (1..focused.len()).rev() { focused.swap(i, (rng.next() % (i as u64 + 1)) as usize) }
        focused.truncate(cap);
        eprintln!("topic slice {p}: {} focus words; {} focus sentences kept (cap {cap})", f.len(), focused.len());
        sources.push(("focus", focused));
    }
    for (name, s) in &sources {
        eprintln!("  {name}: {} sentences, {} words", s.len(), s.iter().map(Vec::len).sum::<usize>());
    }
    let sentences: Vec<Vec<String>> = sources.into_iter().flat_map(|(_, s)| s).collect();

    // Vocabulary and config (kept on resume).
    std::fs::create_dir_all(&out)?;
    if !(resume && out.join("vocab.txt").exists()) {
        let mut counts: HashMap<String, u64> = HashMap::new();
        for w in sentences.iter().flatten() { *counts.entry(w.to_lowercase()).or_default() += 1 }
        let t = std::time::Instant::now();
        let vocab = wordpiece::train(&counts, vocab_size, 3);
        eprintln!("vocabulary: {} pieces from {} distinct words ({:.0?})", vocab.len(), counts.len(), t.elapsed());
        std::fs::write(out.join("vocab.txt"), vocab.join("\n") + "\n")?;
        let config = serde_json::json!({
            "model_type": "bert", "hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4,
            "intermediate_size": 1024, "max_position_embeddings": MAX_LEN, "vocab_size": vocab.len(),
            "layer_norm_eps": 1e-12, "hidden_act": "gelu", "type_vocab_size": 2, "pad_token_id": 0,
            "pre_layer_norm": !args.iter().any(|a| a == "--post-ln"),
            "trained_on": "standards: Wiktionary + WordNet examples, EWT train, Tatoeba, Simple English Wikipedia; SemCor meanings (joint)"
        });
        std::fs::write(out.join("config.json"), serde_json::to_string_pretty(&config)?)?;
    }
    let (bert, vars) = if resume && out.join("model.safetensors").exists() {
        eprintln!("resuming from {}", out.display());
        Bert::load_trainable(&out, dev.clone())?
    } else {
        Bert::new_random(&out, dev.clone())?
    };
    let encode = |s: &Vec<String>| { let (ids, first) = bert.encode_words(s); Encoded { first: first.into_iter().filter(|&f| f < ids.len()).map(|f| f as u16).collect(), ids } };
    let data: Vec<Encoded> = sentences.iter().map(encode).collect();
    let held: Vec<Encoded> = held.iter().map(encode).collect();
    drop(sentences);
    let pieces: usize = data.iter().map(|e| e.ids.len()).sum();
    let unk = bert.special("[UNK]");
    let unknown = data.iter().flat_map(|e| &e.ids).filter(|&&i| i == unk).count();
    eprintln!("corpus: {} sentences, {pieces} pieces ({:.2}% [UNK]); ~{:.1} passes over {steps} steps",
              data.len(), 100.0 * unknown as f64 / pieces as f64, (steps * batch_size) as f64 / data.len() as f64);

    let dict = if joint > 0.0 { dictionary_examples(&bert, &lex, SEMCOR, Path::new("data/oewn"))? } else { vec![] };
    let (dict_train, dict_held) = dict.split_at(dict.len().saturating_sub(1000));
    eprintln!("dictionary objective: weight {joint}, {} SemCor words with synonyms ({} held out)", dict_train.len(), dict_held.len());

    let evaluate = || -> candle_core::Result<(f32, f32)> {
        let mut rng = Rng(99);
        let refs: Vec<&Encoded> = held.iter().collect();
        let chunks: Vec<&[&Encoded]> = refs.chunks(batch_size).collect();
        let mut m = 0.0;
        for c in &chunks {
            let (rows, at, gold) = mask_whole_words(&bert, c, &mut rng);
            m += masked_loss(&bert, &rows, &at, &gold, &dev)?.to_scalar::<f32>()?;
        }
        let mut d = 0.0;
        let drefs: Vec<&DictExample> = dict_held.iter().collect();
        let dchunks: Vec<&[&DictExample]> = drefs.chunks(batch_size).collect();
        for c in &dchunks { d += dictionary_loss(&bert, c, &dev)?.to_scalar::<f32>()? }
        Ok((m / chunks.len() as f32, if dchunks.is_empty() { 0.0 } else { d / dchunks.len() as f32 }))
    };
    let (m0, d0) = evaluate()?;
    let mut best = m0 + joint * d0;
    eprintln!("held-out before training: masked words {m0:.3}, dictionary {d0:.3}");

    let mut opt = AdamW::new(vars.all_vars(), ParamsAdamW { lr: LR, weight_decay: 0.01, ..Default::default() })?;
    let mut rng = Rng(0x2545F4914F6CDD1D);
    let t0 = std::time::Instant::now();
    // Length buckets: sentences sorted by length, cut into batches, batch order
    // shuffled each pass, so padding stays small (lengths range ~5 to 128 pieces).
    let mut by_length: Vec<usize> = (0..data.len()).collect();
    by_length.sort_by_key(|&i| data[i].ids.len());
    let mut batches: Vec<Vec<usize>> = by_length.chunks(batch_size).map(|c| c.to_vec()).collect();
    let mut next_batch = batches.len();
    // Training loss stays on the GPU until a checkpoint (no per-step sync).
    let mut window: Option<candle_core::Tensor> = None;
    for step in 0..steps {
        let lr = if step < WARMUP { LR * (step + 1) as f64 / WARMUP as f64 } else { LR * (steps - step) as f64 / (steps - WARMUP).max(1) as f64 };
        opt.set_learning_rate(lr);
        if next_batch == batches.len() {
            for i in (1..batches.len()).rev() { batches.swap(i, (rng.next() % (i as u64 + 1)) as usize) }
            next_batch = 0;
        }
        let batch: Vec<&Encoded> = batches[next_batch].iter().map(|&i| &data[i]).collect();
        next_batch += 1;
        let (rows, at, gold) = mask_whole_words(&bert, &batch, &mut rng);
        if at.is_empty() { continue }
        let mut loss = masked_loss(&bert, &rows, &at, &gold, &dev)?;
        if joint > 0.0 && !dict_train.is_empty() {
            let b: Vec<&DictExample> = (0..batch_size / 2).map(|_| &dict_train[(rng.next() % dict_train.len() as u64) as usize]).collect();
            loss = (loss + (dictionary_loss(&bert, &b, &dev)? * joint as f64)?)?;
        }
        let l = loss.detach();
        window = Some(match window.take() { Some(w) => (w + l)?, None => l });
        opt.backward_step(&loss)?;
        if (step + 1) % every == 0 || step + 1 == steps {
            let (m, d) = evaluate()?;
            let h = m + joint * d;
            let saved = if h < best { best = h; vars.save(out.join("model.safetensors"))?; " saved" } else { "" };
            let span = if (step + 1) % every == 0 { every as f32 } else { ((step + 1) % every) as f32 };
            let train = window.take().map(|w| w.to_scalar::<f32>()).transpose()?.unwrap_or(0.0) / span;
            eprintln!("  step {:>7}  train {train:.3}  held-out masked {m:.3} dictionary {d:.3}  ({:.0?}){saved}", step + 1, t0.elapsed());
        }
    }
    eprintln!("best held-out {best:.3}; model in {}", out.display());
    Ok(())
}
