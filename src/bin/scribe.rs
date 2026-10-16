//! scribe build-lexicon <oewn-json-dir> <models-dir>
//! scribe train <ewt-dir> <models-dir>       taggers, parser, inflection, fluency model
//! scribe frames <models-dir>                stdin → one JSON frame per clause
//! scribe variants <models-dir> [limit]      stdin → one JSON variant per line
//! scribe train-usage <data-dir> <models-dir> [steps]   usage model + sense index (build with --features metal)

use scribe::{conllu, inflect::Inflector, lm::LanguageModel, parser::Parser, tagger::{Column, Tagger}, text, Engine, Lexicon};
use std::io::Read;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize| args.get(i).map(String::as_str).ok_or("missing argument");
    match args.get(1).map(String::as_str) {
        Some("build-lexicon") => {
            let t = Instant::now();
            let lex = Lexicon::from_oewn_json(Path::new(arg(2)?))?;
            let out = Path::new(arg(3)?).join("lexicon.bin");
            lex.save(&out)?;
            eprintln!(
                "lexicon: {} lemmas, {} synsets, {} categories, {} irregular forms → {:.1} MB in {:.1?}",
                lex.lemmas.len(), lex.synsets.len(), lex.categories.len(), lex.forms.len(), std::fs::metadata(&out)?.len() as f64 / 1e6, t.elapsed()
            );
        }
        Some("train") => {
            let dir = Path::new(arg(2)?);
            let out = Path::new(arg(3)?);
            let read = |s: &str| -> Result<_, Box<dyn std::error::Error>> {
                Ok(conllu::parse(&std::fs::read_to_string(dir.join(format!("en_ewt-ud-{s}.conllu")))?))
            };
            let (train, dev, test) = (read("train")?, read("dev")?, read("test")?);
            let lex = Lexicon::load(&out.join("lexicon.bin"))?;
            eprintln!("EWT: {} train / {} dev / {} test sentences", train.len(), dev.len(), test.len());

            let t = Instant::now();
            let plain = Tagger::train(&train, Column::Upos, None, 5);
            let plain_dev = plain.accuracy(&dev, Column::Upos, None);
            let tagger = Tagger::train(&train, Column::Upos, Some(&lex), 5);
            let (dev_acc, test_acc) = (tagger.accuracy(&dev, Column::Upos, Some(&lex)), tagger.accuracy(&test, Column::Upos, Some(&lex)));
            eprintln!("UPOS dev: {plain_dev:.2}% plain → {dev_acc:.2}% with dictionary; test {test_acc:.2}% ({:.1?})", t.elapsed());
            tagger.save(&out.join("tagger.bin"))?;

            let xpos = Tagger::train(&train, Column::Xpos, Some(&lex), 5);
            eprintln!("XPOS test: {:.2}%", xpos.accuracy(&test, Column::Xpos, Some(&lex)));
            xpos.save(&out.join("xpos.bin"))?;

            let t = Instant::now();
            let mut train_tags = vec![Vec::new(); train.len()];
            for fold in 0..5 {
                let rest: Vec<_> = train.iter().enumerate().filter(|(i, _)| i % 5 != fold).map(|(_, s)| s.clone()).collect();
                let fold_tagger = Tagger::train(&rest, Column::Upos, Some(&lex), 5);
                for (i, s) in train.iter().enumerate().filter(|(i, _)| i % 5 == fold) {
                    let words: Vec<&str> = s.iter().map(|t| t.form.as_str()).collect();
                    train_tags[i] = fold_tagger.tag(&words, Some(&lex)).into_iter().map(String::from).collect();
                }
            }
            let parser = Parser::train(&train, &train_tags, 12);
            let tag = |w: &[&str]| tagger.tag(w, Some(&lex)).into_iter().map(String::from).collect::<Vec<_>>();
            let (uas, las) = parser.evaluate(&test, &tag);
            eprintln!("parser test (predicted tags): UAS {uas:.2}%  LAS {las:.2}% ({:.1?})", t.elapsed());
            parser.save(&out.join("parser.bin"))?;

            let inflector = Inflector::train(&train, &lex);
            inflector.save(&out.join("inflect.bin"))?;

            // Fluency model: usage sentences only (never definitions).
            let treebank: Vec<Vec<String>> = train.iter().map(|s| s.iter().map(|t| t.form.clone()).collect()).collect();
            let wiktionary: Vec<String> = std::fs::read_to_string(dir.join("wiktionary/examples.txt")).map(|t| t.lines().map(String::from).collect()).unwrap_or_default();
            let corpus = scribe::usage::corpus::usage_sentences(&lex, &treebank, &wiktionary);
            let mut lm = LanguageModel::train(&corpus);
            lm.prune(2); // phone-sized: drop phrases seen once
            lm.save(&out.join("lm.bin"))?;
            let held: Vec<Vec<String>> = test.iter().map(|s| s.iter().map(|t| t.form.to_lowercase()).collect()).collect();
            let mean: f64 = held.iter().map(|s| lm.score(s)).sum::<f64>() / held.len() as f64;
            eprintln!("fluency model: {} usage sentences (no definitions); held-out perplexity {:.0}", corpus.len(), (-mean).exp());

            for f in ["lexicon", "tagger", "xpos", "parser", "inflect", "lm"] {
                eprintln!("  {f}.bin {:.1} MB", std::fs::metadata(out.join(format!("{f}.bin")))?.len() as f64 / 1e6);
            }
        }
        Some("frames") => {
            let engine = Engine::load(Path::new(arg(2)?))?;
            let mut input = String::new();
            std::io::stdin().read_to_string(&mut input)?;
            for frame in engine.frames(&input) {
                println!("{}", serde_json::to_string(&frame)?);
            }
        }
        Some("variants") => {
            let t = Instant::now();
            let engine = Engine::load(Path::new(arg(2)?))?;
            let loaded = t.elapsed();
            let limit = args.get(3).and_then(|l| l.parse().ok()).unwrap_or(40);
            let mut input = String::new();
            std::io::stdin().read_to_string(&mut input)?;
            let t = Instant::now();
            let variants = engine.variants(input.trim(), limit);
            eprintln!("{} variants in {:.1?} (engine load {:.1?})", variants.len(), t.elapsed(), loaded);
            for v in variants {
                println!("{}", serde_json::to_string(&v)?);
            }
        }
        Some("train-usage") => train_usage(Path::new(arg(2)?), Path::new(arg(3)?), args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20_000))?,
        _ => eprintln!("usage: scribe build-lexicon | train | frames | variants | train-usage (see source header)"),
    }
    Ok(())
}

fn train_usage(data: &Path, models: &Path, steps: usize) -> Result<(), Box<dyn std::error::Error>> {
    use scribe::usage::{corpus, model::{Config, Device, TrainOptions, UsageModel}, senses::SenseIndex, vocab::Vocab};
    use std::io::{BufRead, Write};

    let lex = Lexicon::load(&models.join("lexicon.bin"))?;
    let tokenize = |s: &str| -> Vec<String> { text::tokenize(s).into_iter().map(|w| w.to_lowercase()).collect() };

    // Wiktionary usage examples (extracted once, cached).
    let cache = data.join("wiktionary/examples.txt");
    if !cache.exists() {
        let t = Instant::now();
        let src = std::io::BufReader::new(std::fs::File::open(data.join("wiktionary/kaikki-English.jsonl"))?);
        let mut out = std::io::BufWriter::new(std::fs::File::create(&cache)?);
        let mut n = 0usize;
        for line in src.lines() {
            for ex in corpus::wiktionary_examples(&line?, 4, 40) {
                writeln!(out, "{}", ex.replace('\n', " "))?;
                n += 1;
            }
        }
        eprintln!("wiktionary: {n} English usage examples extracted in {:.1?}", t.elapsed());
    }
    let mut sentences: Vec<Vec<String>> = std::fs::read_to_string(&cache)?.lines().map(tokenize).collect();
    let wiktionary = sentences.len();

    let ewt = conllu::parse(&std::fs::read_to_string(data.join("en_ewt-ud-train.conllu"))?);
    sentences.extend(ewt.iter().map(|s| s.iter().map(|t| t.form.to_lowercase()).collect()));
    let examples = corpus::wordnet_examples(&lex);
    let (held, train_ex): (Vec<_>, Vec<_>) = examples.into_iter().partition(|e| corpus::is_held_out(e.synset, &e.sentence, 10));
    sentences.extend(train_ex.iter().map(|e| tokenize(&e.sentence)));
    let tokens: usize = sentences.iter().map(Vec::len).sum();
    eprintln!(
        "corpus: {} sentences, {} tokens (wiktionary {wiktionary}, treebank {}, wordnet {} train / {} held out)",
        sentences.len(), tokens, ewt.len(), train_ex.len(), held.len()
    );

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    let config = Config { vocab_size: 0, dim: 256, layers: 4, heads: 4, ff: 1024, max_len: 48 };
    let ckpt = models.join("usage-checkpoint");
    let (mut model, start) = match UsageModel::checkpoint_step(&ckpt) {
        Some(step) if step < steps => {
            eprintln!("resuming from checkpoint at step {step}");
            UsageModel::resume(&ckpt, device)?
        }
        _ => (UsageModel::new_random(Vocab::build(&sentences, 3, 32_000), config, device, 7)?, 0),
    };
    let config = model.config;
    let ids: Vec<Vec<u32>> = sentences.iter().map(|s| model.vocab.encode(s)).collect();
    eprintln!("vocab {} words; model {}×{} layers; training {steps} steps", model.vocab.len(), config.dim, config.layers);

    let t = Instant::now();
    let mut window = Vec::new();
    let opts = TrainOptions { steps, batch: 64, lr: 5e-4, warmup: 1000, seed: 7, start_step: start, checkpoint: Some((ckpt.clone(), 1000)) };
    model.train(&ids, &opts, |step, loss| {
        window.push(loss);
        if (step + 1) % 1000 == 0 {
            let avg = window.iter().sum::<f32>() / window.len() as f32;
            eprintln!("  step {:>6}  loss {avg:.3}  ({:.0?})", step + 1, t.elapsed());
            window.clear();
        }
    })?;
    let out = models.join("usage");
    model.save(&out)?;

    // Sense index from the training examples.
    let locate = |sentence: &str, word: &str| -> Option<(Vec<String>, usize)> {
        if word.contains(' ') {
            return None;
        }
        let toks = tokenize(sentence);
        let at = corpus::find_word(&toks, word, |w| lex.lemmatize(w, None))?;
        (at < config.max_len).then_some((toks, at))
    };
    let mut index = SenseIndex::default();
    let located: Vec<(u32, Vec<String>, usize)> = train_ex.iter().filter_map(|e| locate(&e.sentence, &e.word).map(|(t, a)| (e.synset, t, a))).collect();
    for chunk in located.chunks(256) {
        let toks: Vec<Vec<String>> = chunk.iter().map(|(_, t, _)| t.clone()).collect();
        for ((synset, _, at), vectors) in chunk.iter().zip(model.contextual_batch(&toks)?) {
            index.add(*synset, &vectors[*at]);
        }
    }
    std::fs::write(out.join("senses.bin"), postcard::to_allocvec(&index)?)?;
    eprintln!("sense index: {} senses with examples", index.len());

    // Held-out evaluation, same cases for every method.
    let lesk = scribe::wsd::Disambiguator::new(&lex, false);
    let cases: Vec<(&corpus::SenseExample, Vec<String>, usize)> = held
        .iter()
        .filter(|e| lex.senses(&e.word, lex.synsets[e.synset as usize].pos).len() >= 2)
        .filter_map(|e| locate(&e.sentence, &e.word).map(|(t, a)| (e, t, a)))
        .collect();
    let mut vectors = Vec::with_capacity(cases.len());
    for chunk in cases.chunks(256) {
        let toks: Vec<Vec<String>> = chunk.iter().map(|(_, t, _)| t.clone()).collect();
        for ((_, _, at), v) in chunk.iter().zip(model.contextual_batch(&toks)?) {
            vectors.push(v[*at].clone());
        }
    }
    let names = ["first-listed sense", "definition overlap", "usage model", "usage + listing order"];
    let mut exact = [0usize; 4];
    let mut category = [0usize; 4];
    for ((e, toks, _), v) in cases.iter().zip(&vectors) {
        let pos = lex.synsets[e.synset as usize].pos;
        let senses = lex.senses(&e.word, pos);
        let lemmas: Vec<String> = toks.iter().map(|w| lex.lemmatize(w, None).into_iter().next().unwrap_or(w.clone())).collect();
        let lesk_pick = lesk.disambiguate(&lex, &e.word, pos, &lemmas);
        let usage_pick = index.best(senses, v).or(lesk_pick);
        let blended = senses
            .iter()
            .enumerate()
            .map(|(rank, &s)| (s, index.similarity(s, v).unwrap_or(0.0) + 0.15 / (1.0 + rank as f32)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(s, _)| s);
        let picks = [Some(senses[0]), lesk_pick, usage_pick, blended];
        let cat = |s: u32| lex.synsets[s as usize].category;
        for (i, p) in picks.iter().enumerate() {
            exact[i] += (*p == Some(e.synset)) as usize;
            category[i] += p.is_some_and(|p| cat(p) == cat(e.synset)) as usize;
        }
    }
    let pct = |x: usize| 100.0 * x as f64 / cases.len().max(1) as f64;
    eprintln!("held-out sense test ({} unseen dictionary examples):", cases.len());
    for i in 0..4 {
        eprintln!("  {:<22} exact {:.1}%   category {:.1}%", names[i], pct(exact[i]), pct(category[i]));
    }
    Ok(())
}
