//! ScribeSLM: BERT-mini's weights (volume) adapted to usage standards by
//! continued masked-word training on usage sentences only — Wiktionary and
//! WordNet examples (WordNet's held-out split kept out), the EWT treebank's
//! training sentences, and Tatoeba. No definitions; nothing from the test sets
//! (SWORDS, ALL, EWT dev/test).
//!
//!   cargo run --release --features metal --example slm_adapt [steps] [--from data/bert-mini] [--out models/scribe-slm]
//!
//! Masking as BERT (15% of pieces; 80% [MASK], 10% random, 10% unchanged).
//! Keeps the step with the lowest held-out loss (2,000 Tatoeba sentences
//! never trained on).
//!
//! --joint [w] (output models/scribe-slm-joint): usage and dictionary learned
//! together. Each step also takes a batch of SemCor words (hand-labelled with
//! their WordNet meaning, in base form so synonyms fit grammatically), hides
//! the word, and adds w × −log P(slot ∈ the word or its meaning's synonyms).
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use scribe::bert::Bert;
use scribe::semcor;
use scribe::usage::corpus;
use scribe::{conllu, text, Lexicon};
use std::path::{Path, PathBuf};

const BATCH: usize = 64;
const HELD_OUT: usize = 2000;
const WARMUP: usize = 1000;
const LR: f64 = 1e-4;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

/// Masks a batch: returns the input rows, the masked (row, col) positions and their original ids.
fn mask(bert: &Bert, batch: &[&Vec<u32>], rng: &mut Rng) -> (Vec<Vec<u32>>, Vec<(usize, usize)>, Vec<u32>) {
    let (m, specials) = (bert.special("[MASK]"), [bert.special("[CLS]"), bert.special("[SEP]"), 0]);
    let (mut rows, mut at, mut gold) = (Vec::new(), Vec::new(), Vec::new());
    for (r, ids) in batch.iter().enumerate() {
        let mut row = (*ids).clone();
        for c in 0..row.len() {
            if specials.contains(&row[c]) || rng.unit() >= 0.15 { continue }
            at.push((r, c));
            gold.push(row[c]);
            let x = rng.unit();
            if x < 0.8 { row[c] = m } else if x < 0.9 { row[c] = 1000 + (rng.next() % (bert.vocab_len() as u64 - 1000)) as u32 }
        }
        rows.push(row);
    }
    (rows, at, gold)
}

fn loss(bert: &Bert, rows: &[Vec<u32>], at: &[(usize, usize)], gold: &[u32], dev: &Device) -> candle_core::Result<Tensor> {
    let lp = bert.masked_log_probs(rows, at)?;
    let picked = lp.gather(&Tensor::new(gold, dev)?.unsqueeze(1)?, 1)?;
    picked.neg()?.mean_all()
}

/// A SemCor word whose slot should accept `accept` (the word and its meaning's synonyms, single pieces).
struct DictExample {
    ids: Vec<u32>,
    at: usize,
    accept: Vec<u32>,
}

fn dictionary_examples(bert: &Bert, lex: &Lexicon) -> Result<Vec<DictExample>, Box<dyn std::error::Error>> {
    let base = "data/semcor/wsdeval/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor";
    let sentences = semcor::parse(&std::fs::read_to_string(format!("{base}.data.xml"))?, &std::fs::read_to_string(format!("{base}.gold.key.txt"))?);
    let keys = semcor::sense_keys(Path::new("data/oewn"), lex)?;
    let mut out = Vec::new();
    for s in &sentences {
        let (ids, first) = bert.encode_words(&s.words);
        for inst in &s.instances {
            let word = s.words[inst.index].to_lowercase();
            let Some(&gold) = inst.keys.iter().find_map(|k| keys.get(k)) else { continue };
            let (Some(id), Some(&at)) = (bert.single_piece(&word), first.get(inst.index)) else { continue };
            if word != inst.lemma.to_lowercase() || at >= ids.len() { continue }
            let mut accept = vec![id];
            for m in &lex.synsets[gold as usize].members {
                if let Some(x) = bert.single_piece(m).filter(|_| !m.contains(' ')) {
                    if !accept.contains(&x) { accept.push(x) }
                }
            }
            if accept.len() > 1 {
                out.push(DictExample { ids: ids.clone(), at, accept });
            }
        }
    }
    Ok(out)
}

/// −log of the probability mass the hidden slot puts on its acceptable words.
fn dictionary_loss(bert: &Bert, batch: &[&DictExample], dev: &Device) -> candle_core::Result<Tensor> {
    let m = bert.special("[MASK]");
    let rows: Vec<Vec<u32>> = batch.iter().map(|e| { let mut r = e.ids.clone(); r[e.at] = m; r }).collect();
    let at: Vec<(usize, usize)> = batch.iter().enumerate().map(|(i, e)| (i, e.at)).collect();
    let lp = bert.masked_log_probs(&rows, &at)?; // [n, V]
    let k = batch.iter().map(|e| e.accept.len()).max().unwrap_or(1);
    let (mut idx, mut pad) = (Vec::new(), Vec::new());
    for e in batch {
        for j in 0..k {
            idx.push(*e.accept.get(j).unwrap_or(&e.accept[0]));
            pad.push(if j < e.accept.len() { 0f32 } else { -1e9 });
        }
    }
    let n = batch.len();
    let picked = (lp.gather(&Tensor::from_vec(idx, (n, k), dev)?, 1)? + Tensor::from_vec(pad, (n, k), dev)?)?;
    let max = picked.max_keepdim(1)?;
    let lse = (picked.broadcast_sub(&max)?.exp()?.sum_keepdim(1)?.log()? + max)?;
    lse.neg()?.mean_all()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30_000);
    let flag = |name: &str, default: &str| -> PathBuf {
        PathBuf::from(args.windows(2).find(|w| w[0] == name).map(|w| w[1].as_str()).unwrap_or(default))
    };
    let joint: Option<f64> = args.iter().position(|a| a == "--joint").map(|i| args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1.0));
    let from = flag("--from", "data/bert-mini");
    let out = flag("--out", if joint.is_some() { "models/scribe-slm-joint" } else { "models/scribe-slm" });
    #[cfg(feature = "metal")]
    let dev = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let dev = Device::Cpu;

    // Usage sentences only.
    let lex = Lexicon::load(Path::new("models/lexicon.bin"))?;
    let words = |s: &str| text::tokenize(s);
    let mut sentences: Vec<Vec<String>> = std::fs::read_to_string("data/wiktionary/examples.txt")?.lines().map(words).collect();
    let wiktionary = sentences.len();
    let ewt = conllu::parse(&std::fs::read_to_string("data/en_ewt-ud-train.conllu")?);
    sentences.extend(ewt.iter().map(|s| s.iter().map(|t| t.form.clone()).collect()));
    let examples: Vec<_> = corpus::wordnet_examples(&lex).into_iter().filter(|e| !corpus::is_held_out(e.synset, &e.sentence, 10)).collect();
    sentences.extend(examples.iter().map(|e| words(&e.sentence)));
    let tatoeba: Vec<Vec<String>> = std::fs::read_to_string("data/corpus/tatoeba.txt")?.lines().map(words).collect();
    let (held, tatoeba_train) = tatoeba.split_at(HELD_OUT);
    let n_tatoeba = tatoeba_train.len();
    sentences.extend(tatoeba_train.iter().cloned());
    sentences.retain(|s| !s.is_empty());

    let (bert, vars) = Bert::load_trainable(&from, dev.clone())?;
    let ids: Vec<Vec<u32>> = sentences.iter().map(|s| bert.encode_sentence(s)).collect();
    let held: Vec<Vec<u32>> = held.iter().map(|s| bert.encode_sentence(s)).collect();
    drop(sentences);
    let pieces: usize = ids.iter().map(Vec::len).sum();
    eprintln!("corpus: {} sentences, {pieces} pieces (wiktionary {wiktionary}, treebank {}, wordnet {}, tatoeba {n_tatoeba}); held out {HELD_OUT}",
              ids.len(), ewt.len(), examples.len());

    let held_loss = |rng_seed: u64| -> candle_core::Result<f32> {
        let mut rng = Rng(rng_seed);
        let mut total = 0.0;
        let refs: Vec<&Vec<u32>> = held.iter().collect();
        let chunks: Vec<&[&Vec<u32>]> = refs.chunks(BATCH).collect();
        for c in &chunks {
            let (rows, at, gold) = mask(&bert, c, &mut rng);
            total += loss(&bert, &rows, &at, &gold, &dev)?.to_scalar::<f32>()?;
        }
        Ok(total / chunks.len() as f32)
    };
    // Dictionary examples (joint mode): the last 1,000 held out.
    let dict = match joint { Some(_) => dictionary_examples(&bert, &lex)?, None => vec![] };
    let (dict_train, dict_held) = dict.split_at(dict.len().saturating_sub(1000));
    if joint.is_some() {
        eprintln!("dictionary examples: {} SemCor words with synonyms ({} held out)", dict_train.len(), dict_held.len());
    }
    let dict_held_loss = || -> candle_core::Result<f32> {
        if dict_held.is_empty() { return Ok(0.0) }
        let refs: Vec<&DictExample> = dict_held.iter().collect();
        let chunks: Vec<&[&DictExample]> = refs.chunks(BATCH).collect();
        let mut total = 0.0;
        for c in &chunks { total += dictionary_loss(&bert, c, &dev)?.to_scalar::<f32>()? }
        Ok(total / chunks.len() as f32)
    };
    let w = joint.unwrap_or(0.0) as f32;
    let start = held_loss(99)? + w * dict_held_loss()?;
    eprintln!("held-out loss before adapting: {start:.3} (masked words{})", if joint.is_some() { " + w × dictionary" } else { "" });

    let mut opt = AdamW::new(vars.all_vars(), ParamsAdamW { lr: LR, weight_decay: 0.01, ..Default::default() })?;
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut best = start;
    std::fs::create_dir_all(&out)?;
    let t0 = std::time::Instant::now();
    let mut window = 0.0f32;
    for step in 0..steps {
        let lr = if step < WARMUP { LR * (step + 1) as f64 / WARMUP as f64 } else { LR * (steps - step) as f64 / (steps - WARMUP) as f64 };
        opt.set_learning_rate(lr);
        let batch: Vec<&Vec<u32>> = (0..BATCH).map(|_| &ids[(rng.next() % ids.len() as u64) as usize]).collect();
        let (rows, at, gold) = mask(&bert, &batch, &mut rng);
        if at.is_empty() { continue }
        let mut l = loss(&bert, &rows, &at, &gold, &dev)?;
        if let Some(wt) = joint {
            let b: Vec<&DictExample> = (0..BATCH).map(|_| &dict_train[(rng.next() % dict_train.len() as u64) as usize]).collect();
            l = (l + (dictionary_loss(&bert, &b, &dev)? * wt)?)?;
        }
        window += l.to_scalar::<f32>()?;
        opt.backward_step(&l)?;
        if (step + 1) % 2000 == 0 {
            let d = dict_held_loss()?;
            let h = held_loss(99)? + w * d;
            let saved = if h < best { best = h; vars.save(out.join("model.safetensors"))?; " saved" } else { "" };
            eprintln!("  step {:>6}  train loss {:.3}  held-out {h:.3} (dictionary {d:.3})  ({:.0?}){saved}", step + 1, window / 2000.0, t0.elapsed());
            window = 0.0;
        }
    }
    for f in ["vocab.txt", "config.json"] {
        std::fs::copy(from.join(f), out.join(f))?;
    }
    eprintln!("best held-out loss {best:.3} (from {start:.3}); saved to {}", out.display());
    Ok(())
}
