//! scribe build-lexicon <oewn-json-dir> <models-dir>
//! scribe train <ewt-dir> <models-dir>       taggers, parser, inflection, fluency model
//! scribe frames <models-dir>                stdin → one JSON frame per clause
//! scribe variants <models-dir> [limit]      stdin → one JSON variant per line
//! scribe train-usage <data-dir> <models-dir> [steps] [--morph]   usage model + sense index (build with --features metal);
//!                                                       --morph splits words into base + ending → models/usage-morph;
//!                                                       --extra <file> (repeatable) adds one-sentence-per-line text
//!                                                       (scripts/extract_corpus.py); --name <dir> names the output
//! scribe build-senses <models-dir> [usage-dir-name]     rebuild the sense index from a trained usage model
//! scribe train-senses <data-dir> <models-dir> [steps]   sense modeling: continue models/usage with a supersense
//!                                                       objective (dictionary-allowed sets + SemCor gold) → models/usage-sense;
//!                                                       --control: same corpus and schedule, words only → models/usage-control
//! scribe train-head <data-dir> <models-dir> [usage-dir-name] [--held-out]
//!                                                       sense head on SemCor's sense-tagged text (masked-word vectors from the
//!                                                       usage model) → <usage-dir>/head, which the engine then uses to choose
//!                                                       senses; --held-out trains on even documents only (as semcor_eval does)

use scribe::{conllu, inflect::Inflector, lm::LanguageModel, parser::Parser, tagger::{Column, Tagger}, text, Engine, Lexicon};
use std::io::Read;
use std::path::{Path, PathBuf};
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
        Some("build-senses") => {
            let models = Path::new(arg(2)?);
            #[cfg(feature = "metal")]
            let device = scribe::usage::model::Device::new_metal(0)?;
            #[cfg(not(feature = "metal"))]
            let device = scribe::usage::model::Device::Cpu;
            let name = args.get(3).map(String::as_str).unwrap_or("usage");
            let model = scribe::usage::model::UsageModel::load(&models.join(name), device)?;
            build_senses(&model, &Lexicon::load(&models.join("lexicon.bin"))?, &models.join(name))?;
        }
        Some("train-head") => {
            let name = args.get(4).filter(|a| !a.starts_with("--")).map_or("usage", String::as_str);
            train_head(Path::new(arg(2)?), Path::new(arg(3)?), name, args.iter().any(|a| a == "--held-out"))?
        }
        Some("train-senses") => {
            let steps = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20_000);
            let control = args.iter().any(|a| a == "--control");
            train_senses(Path::new(arg(2)?), Path::new(arg(3)?), steps, control)?
        }
        Some("train-usage") => {
            let steps = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20_000);
            let morph = args.iter().any(|a| a == "--morph");
            let extra: Vec<PathBuf> = args.windows(2).filter(|w| w[0] == "--extra").map(|w| PathBuf::from(&w[1])).collect();
            let name = args.windows(2).find(|w| w[0] == "--name").map(|w| w[1].clone());
            train_usage(Path::new(arg(2)?), Path::new(arg(3)?), steps, morph, &extra, name)?
        }
        _ => eprintln!("usage: scribe build-lexicon | train | frames | variants | train-usage (see source header)"),
    }
    Ok(())
}

/// Sense modeling (after SenseBERT): continue training the usage model with a
/// second objective — name each masked word's supersense. Targets are the
/// supersenses the dictionary allows for the word (context must choose), or
/// SemCor's gold one. Only even-numbered SemCor documents are used; odd ones
/// stay held out for examples/semcor_eval.rs.
fn train_senses(data: &Path, models: &Path, steps: usize, control: bool) -> Result<(), Box<dyn std::error::Error>> {
    use scribe::usage::{corpus, model::{SenseTargets, TrainOptions, UsageModel}};
    use std::collections::BTreeSet;

    let lex = Lexicon::load(&models.join("lexicon.bin"))?;
    let tokenize = |s: &str| -> Vec<String> { text::tokenize(s).into_iter().map(|w| w.to_lowercase()).collect() };
    let mut sentences: Vec<Vec<String>> = std::fs::read_to_string(data.join("wiktionary/examples.txt"))?.lines().map(tokenize).collect();
    let ewt = conllu::parse(&std::fs::read_to_string(data.join("en_ewt-ud-train.conllu"))?);
    sentences.extend(ewt.iter().map(|s| s.iter().map(|t| t.form.to_lowercase()).collect()));
    let examples = corpus::wordnet_examples(&lex);
    sentences.extend(examples.iter().filter(|e| !corpus::is_held_out(e.synset, &e.sentence, 10)).map(|e| tokenize(&e.sentence)));
    let mut gold: Vec<Vec<Option<u16>>> = vec![Vec::new(); sentences.len()];

    let wsd = data.join("semcor/wsdeval/WSD_Evaluation_Framework/Training_Corpora/SemCor");
    let semcor = scribe::semcor::parse(&std::fs::read_to_string(wsd.join("semcor.data.xml"))?, &std::fs::read_to_string(wsd.join("semcor.gold.key.txt"))?);
    let keys = scribe::semcor::sense_keys(&data.join("oewn"), &lex)?;
    let docs: Vec<&str> = semcor.iter().map(|s| s.document.as_str()).collect::<BTreeSet<_>>().into_iter().collect();
    let held: BTreeSet<&str> = docs.iter().skip(1).step_by(2).copied().collect();
    let mut tagged = 0;
    for s in semcor.iter().filter(|s| !held.contains(s.document.as_str())) {
        let mut g = vec![None; s.words.len()];
        for inst in &s.instances {
            if let Some(&synset) = inst.keys.iter().find_map(|k| keys.get(k)) {
                g[inst.index] = Some(lex.synsets[synset as usize].category);
                tagged += 1;
            }
        }
        sentences.push(s.words.iter().map(|w| w.to_lowercase()).collect());
        gold.push(g);
    }
    eprintln!("corpus: {} sentences ({} SemCor training sentences, {tagged} gold supersenses)", sentences.len(), semcor.len() - semcor.iter().filter(|s| held.contains(s.document.as_str())).count());

    #[cfg(feature = "metal")]
    let device = scribe::usage::model::Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = scribe::usage::model::Device::Cpu;
    // The control isolates the supersense objective: identical corpus, steps,
    // learning rate and seed, masked words only.
    let name = if control { "usage-control" } else { "usage-sense" };
    let ckpt = models.join(format!("{name}-checkpoint"));
    let (mut model, start) = match UsageModel::checkpoint_step(&ckpt) {
        Some(step) if step < steps => {
            eprintln!("resuming from checkpoint at step {step}");
            UsageModel::resume(&ckpt, device)?
        }
        _ if control => (UsageModel::load(&models.join("usage"), device)?, 0),
        _ => (UsageModel::load(&models.join("usage"), device)?.with_supersenses(lex.categories.len())?, 0),
    };

    // Allowed supersenses per vocabulary word, from every sense of every
    // lemma it can be. The 100 most frequent words (mostly function words,
    // whose WordNet entries are rare nouns like "a" = vitamin A) get none.
    let by_id: Vec<Vec<u16>> = (0..model.vocab.len() as u32)
        .map(|id| {
            if (id as usize) < scribe::usage::vocab::SPECIALS.len() + 100 {
                return vec![];
            }
            let mut cats: Vec<u16> = lex
                .lemmatize(model.vocab.word(id), None)
                .iter()
                .filter_map(|l| lex.lemmas.get(l))
                .flatten()
                .flat_map(|(_, senses)| senses.iter().map(|&s| lex.synsets[s as usize].category))
                .collect();
            cats.sort();
            cats.dedup();
            cats
        })
        .collect();
    eprintln!("{} of {} vocabulary words have dictionary supersenses", by_id.iter().filter(|c| !c.is_empty()).count(), by_id.len());
    let ids: Vec<Vec<u32>> = sentences.iter().map(|s| model.vocab.encode(s)).collect();
    let targets = SenseTargets { by_id, gold, weight: 1.0 };

    let t = Instant::now();
    let (mut words_avg, mut senses_avg) = (Vec::new(), Vec::new());
    let opts = TrainOptions { steps, batch: 64, lr: 3e-4, warmup: 500, seed: 11, start_step: start, checkpoint: Some((ckpt.clone(), 1000)) };
    model.train_with_senses(&ids, &opts, (!control).then_some(&targets), |step, w, s| {
        words_avg.push(w);
        senses_avg.push(s);
        if (step + 1) % 1000 == 0 {
            let avg = |v: &Vec<f32>| v.iter().sum::<f32>() / v.len() as f32;
            eprintln!("  step {:>6}  word loss {:.3}  supersense loss {:.3}  ({:.0?})", step + 1, avg(&words_avg), avg(&senses_avg), t.elapsed());
            words_avg.clear();
            senses_avg.clear();
        }
    })?;
    let out = models.join(name);
    model.save(&out)?;
    build_senses(&model, &lex, &out)?;
    Ok(())
}

fn train_usage(data: &Path, models: &Path, steps: usize, morph: bool, extra: &[PathBuf], name: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    use scribe::usage::{corpus, model::{Config, Device, TrainOptions, UsageModel}, vocab::Vocab};
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
    for file in extra {
        let more = corpus::plain_sentences(&std::fs::read_to_string(file)?, 40);
        eprintln!("extra corpus {}: {} sentences", file.display(), more.len());
        sentences.extend(more);
    }
    let tokens: usize = sentences.iter().map(Vec::len).sum();
    eprintln!(
        "corpus: {} sentences, {} tokens (wiktionary {wiktionary}, treebank {}, wordnet {} train / {} held out)",
        sentences.len(), tokens, ewt.len(), train_ex.len(), held.len()
    );

    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    let name = name.unwrap_or_else(|| if morph { "usage-morph" } else { "usage" }.to_string());
    let name = name.as_str();
    // Base + ending pieces: split table built from every word in the corpus.
    let morphology = morph.then(|| {
        scribe::usage::morph::Morphology::build(sentences.iter().flatten().cloned(), |w| lex.lemmatize(w, None))
    });
    // Without morphology the pieces are the words: move, don't copy (the
    // corpus can be tens of millions of tokens).
    let pieces: Vec<Vec<String>> = match &morphology {
        Some(m) => sentences.iter().map(|s| m.split_sentence(s).0).collect(),
        None => std::mem::take(&mut sentences),
    };
    drop(sentences);
    if let Some(m) = &morphology {
        eprintln!("morphology: {} inflected word forms split into base + ending", m.len());
    }
    let config = Config { vocab_size: 0, dim: 256, layers: 4, heads: 4, ff: 1024, max_len: 48 };
    let ckpt = models.join(format!("{name}-checkpoint"));
    let (mut model, start) = match UsageModel::checkpoint_step(&ckpt) {
        Some(step) if step < steps => {
            eprintln!("resuming from checkpoint at step {step}");
            UsageModel::resume(&ckpt, device)?
        }
        _ => {
            let fresh = UsageModel::new_random(Vocab::build(&pieces, 3, 32_000), config, device, 7)?;
            (match morphology { Some(m) => fresh.with_morphology(m), None => fresh }, 0)
        }
    };
    let config = model.config;
    let ids: Vec<Vec<u32>> = pieces.iter().map(|s| model.vocab.encode(s)).collect();
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
    let out = models.join(name);
    model.save(&out)?;

    build_senses(&model, &lex, &out)?;
    Ok(())
}

/// Sense index from WordNet example sentences (training split), using the
/// context vector with the word hidden, centered on the mean; then the
/// held-out evaluation against first-listed and definition-overlap baselines.
fn build_senses(model: &scribe::usage::model::UsageModel, lex: &Lexicon, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use scribe::engine::choose_sense;
    use scribe::usage::{corpus, senses::SenseIndex};
    let tokenize = |s: &str| -> Vec<String> { text::tokenize(s).into_iter().map(|w| w.to_lowercase()).collect() };
    let max = model.config.max_len;
    let examples = corpus::wordnet_examples(lex);
    let located: Vec<(bool, &corpus::SenseExample, Vec<String>, usize)> = examples
        .iter()
        .filter(|e| !e.word.contains(' '))
        .filter_map(|e| {
            let toks = tokenize(&e.sentence);
            let at = corpus::find_word(&toks, &e.word, |w| lex.lemmatize(w, None))?;
            (at < max).then(|| (corpus::is_held_out(e.synset, &e.sentence, 10), e, toks, at))
        })
        .collect();
    // With base+ending pieces a long sentence can overflow the model's window,
    // leaving later words without a vector: those examples are skipped.
    let mut vectors: Vec<Option<Vec<f32>>> = Vec::with_capacity(located.len());
    for chunk in located.chunks(256) {
        let masked: Vec<Vec<String>> = chunk.iter().map(|(_, _, t, at)| { let mut t = t.clone(); t[*at] = "[MASK]".into(); t }).collect();
        for ((_, _, _, at), h) in chunk.iter().zip(model.contextual_batch(&masked)?) {
            vectors.push(h.get(*at).cloned());
        }
    }
    let (located, vectors): (Vec<_>, Vec<Vec<f32>>) =
        located.into_iter().zip(vectors).filter_map(|(l, v)| v.map(|v| (l, v))).unzip();
    let train: Vec<usize> = (0..located.len()).filter(|&i| !located[i].0).collect();
    let dim = vectors.first().map_or(0, Vec::len);
    let mut center = vec![0.0f32; dim];
    for &i in &train {
        for (c, x) in center.iter_mut().zip(&vectors[i]) {
            *c += x / train.len() as f32;
        }
    }
    let mut index = SenseIndex::with_center(center);
    for &i in &train {
        index.add(located[i].1.synset, &vectors[i]);
    }
    std::fs::write(out.join("senses.bin"), postcard::to_allocvec(&index)?)?;
    index.build_backoff(lex);
    eprintln!("sense index: {} senses with examples", index.len());

    // Held-out evaluation. The override margin is tuned on even-numbered cases
    // and reported on odd ones, with and without the LMMS-style backoff.
    let lesk = scribe::wsd::Disambiguator::new(lex, false);
    let without_backoff = |s: u32, v: &[f32]| index.similarity(s, v).map(|x| (x, scribe::usage::senses::Evidence::Examples));
    let cases: Vec<(&corpus::SenseExample, &Vec<String>, &Vec<f32>)> = located
        .iter()
        .zip(&vectors)
        .filter(|((held, e, _, _), _)| *held && lex.senses(&e.word, lex.synsets[e.synset as usize].pos).len() >= 2)
        .map(|((_, e, t, _), v)| (*e, t, v))
        .collect();
    let (tune, report): (Vec<_>, Vec<_>) = cases.iter().enumerate().partition(|(i, _)| i % 2 == 0);
    let accuracy = |set: &[(usize, &(&corpus::SenseExample, &Vec<String>, &Vec<f32>))], margin: f32, backoff: bool| -> (f64, f64) {
        let (mut exact, mut cat) = (0usize, 0usize);
        for (_, (e, _, v)) in set {
            let senses = lex.senses(&e.word, lex.synsets[e.synset as usize].pos);
            let pick = if backoff {
                choose_sense(senses, |s| index.score(s, v, lex), margin)
            } else {
                choose_sense(senses, |s| without_backoff(s, v), margin)
            }
            .map(|(s, _)| s);
            exact += (pick == Some(e.synset)) as usize;
            cat += pick.is_some_and(|p| lex.synsets[p as usize].category == lex.synsets[e.synset as usize].category) as usize;
        }
        (100.0 * exact as f64 / set.len().max(1) as f64, 100.0 * cat as f64 / set.len().max(1) as f64)
    };
    let margins = [0.0f32, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7];
    let (mut first, mut first_cat, mut overlap) = (0usize, 0usize, 0usize);
    for (_, (e, t, _)) in &report {
        let pos = lex.synsets[e.synset as usize].pos;
        let senses = lex.senses(&e.word, pos);
        let lemmas: Vec<String> = t.iter().map(|w| lex.lemmatize(w, None).into_iter().next().unwrap_or(w.clone())).collect();
        first += (senses[0] == e.synset) as usize;
        first_cat += (lex.synsets[senses[0] as usize].category == lex.synsets[e.synset as usize].category) as usize;
        overlap += (lesk.disambiguate(lex, &e.word, pos, &lemmas) == Some(e.synset)) as usize;
    }
    let pct = |x: usize| 100.0 * x as f64 / report.len().max(1) as f64;
    eprintln!("held-out sense test: tune on {} / report on {} unseen dictionary examples", tune.len(), report.len());
    eprintln!("  first-listed sense             exact {:.1}%   category {:.1}%", pct(first), pct(first_cat));
    eprintln!("  definition overlap (baseline)  exact {:.1}%", pct(overlap));
    for m in margins {
        let (e, c) = accuracy(&report, m, true);
        eprintln!("    margin {m:<4} + backoff: exact {e:.1}%  category {c:.1}%");
    }
    for backoff in [false, true] {
        let best = *margins.iter().max_by(|a, b| accuracy(&tune, **a, backoff).0.total_cmp(&accuracy(&tune, **b, backoff).0)).unwrap();
        let (exact, cat) = accuracy(&report, best, backoff);
        eprintln!("  usage {:<24} exact {exact:.1}%   category {cat:.1}%   (margin {best})", if backoff { "+ relatives/category" } else { "examples only" });
    }
    Ok(())
}

/// Trains the sense head on SemCor: for every sense-tagged word with more than
/// one dictionary sense, the usage model's vector at the masked word, its
/// candidate senses in dictionary order, and the gold sense.
fn train_head(data: &Path, models: &Path, usage_name: &str, held_out: bool) -> Result<(), Box<dyn std::error::Error>> {
    use scribe::usage::{model::{Device, UsageModel}, sense_head::{self, Candidate, SenseHead}};
    use std::collections::BTreeSet;

    let t = Instant::now();
    let lex = Lexicon::load(&models.join("lexicon.bin"))?;
    let dir = models.join(usage_name);
    let usage = UsageModel::load(&dir, Device::Cpu)?;
    let wsd = data.join("semcor/wsdeval/WSD_Evaluation_Framework/Training_Corpora/SemCor");
    let semcor = scribe::semcor::parse(&std::fs::read_to_string(wsd.join("semcor.data.xml"))?, &std::fs::read_to_string(wsd.join("semcor.gold.key.txt"))?);
    let keys = scribe::semcor::sense_keys(&data.join("oewn"), &lex)?;
    let docs: Vec<&str> = semcor.iter().map(|s| s.document.as_str()).collect::<BTreeSet<_>>().into_iter().collect();
    let held: BTreeSet<&str> = if held_out { docs.iter().skip(1).step_by(2).copied().collect() } else { BTreeSet::new() };
    let mut examples = Vec::new();
    let sentences: Vec<_> = semcor.iter().filter(|s| !held.contains(s.document.as_str())).collect();
    for (n, s) in sentences.iter().enumerate() {
        if n % 2000 == 0 { eprintln!("  {n}/{} sentences, {} examples", sentences.len(), examples.len()) }
        let lower: Vec<String> = s.words.iter().map(|w| w.to_lowercase()).collect();
        let masked: Vec<Vec<String>> = s.instances.iter().map(|i| { let mut m = lower.clone(); m[i.index] = "[MASK]".into(); m }).collect();
        let Ok(batch) = usage.contextual_batch(&masked) else { continue };
        for (inst, h) in s.instances.iter().zip(batch) {
            let senses = lex.senses(&inst.lemma, inst.pos);
            let Some(gold) = inst.keys.iter().filter_map(|k| keys.get(k)).find_map(|g| senses.iter().position(|s| s == g)) else { continue };
            let Some(vector) = h.get(inst.index).cloned() else { continue };
            if senses.len() < 2 { continue }
            let candidates = senses.iter().enumerate().map(|(rank, &x)| Candidate { synset: x, category: lex.synsets[x as usize].category, rank }).collect();
            examples.push(sense_head::Example { vector, candidates, gold, context: vec![] });
        }
    }
    eprintln!("{} training examples from {} sentences{}", examples.len(), sentences.len(), if held_out { " (even documents)" } else { "" });
    let head = SenseHead::train(&examples, lex.categories.len(), &sense_head::TrainOptions::default(), Device::Cpu)?;
    let out = dir.join("head");
    head.save(&out)?;
    let bytes: u64 = ["senses.safetensors", "senses.json"].iter().filter_map(|f| std::fs::metadata(out.join(f)).ok()).map(|m| m.len()).sum();
    eprintln!("sense head → {} ({:.1} MB) in {:.1?}", out.display(), bytes as f64 / 1e6, t.elapsed());
    Ok(())
}
