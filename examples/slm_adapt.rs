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
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use scribe::bert::Bert;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(30_000);
    let flag = |name: &str, default: &str| -> PathBuf {
        PathBuf::from(args.windows(2).find(|w| w[0] == name).map(|w| w[1].as_str()).unwrap_or(default))
    };
    let (from, out) = (flag("--from", "data/bert-mini"), flag("--out", "models/scribe-slm"));
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
    let start = held_loss(99)?;
    eprintln!("held-out masked-word loss before adapting: {start:.3}");

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
        let l = loss(&bert, &rows, &at, &gold, &dev)?;
        window += l.to_scalar::<f32>()?;
        opt.backward_step(&l)?;
        if (step + 1) % 2000 == 0 {
            let h = held_loss(99)?;
            let saved = if h < best { best = h; vars.save(out.join("model.safetensors"))?; " saved" } else { "" };
            eprintln!("  step {:>6}  train loss {:.3}  held-out {h:.3}  ({:.0?}){saved}", step + 1, window / 2000.0, t0.elapsed());
            window = 0.0;
        }
    }
    for f in ["vocab.txt", "config.json"] {
        std::fs::copy(from.join(f), out.join(f))?;
    }
    eprintln!("best held-out loss {best:.3} (from {start:.3}); saved to {}", out.display());
    Ok(())
}
