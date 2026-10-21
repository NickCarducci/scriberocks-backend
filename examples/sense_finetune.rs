//! Meaning-picking with the usage model fine-tuned together with the sense
//! head (the shipped head scores fixed vectors). Trained on SemCor, scored on
//! the ALL test set of the Word Sense Disambiguation Evaluation Framework.
//!
//!   cargo run --release --features metal --example sense_finetune [usage-dir] [epochs] [--enc-lr X] [--gloss]
//!
//! Per candidate sense: h · (synset vector + category vector) + rank bias,
//! as SenseHead, with h the fine-tuned vector of the (visible) word. A sense
//! overrides the first-listed one only past a margin; ALL is reported at
//! several margins. Shipped head (fixed vectors, margin 3): 62.4%; first-listed 55.6%.
//!
//! --gloss adds a gloss vector to each sense: the same encoder's mean vector
//! over the synset's first members and its definition (a bi-encoder, as in
//! Blevins & Zettlemoyer 2020), so senses unseen in SemCor aren't left on the
//! shared fallback row. Definitions describe senses here; they never enter the
//! usage model's training sentences (src/usage/corpus.rs).
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, AdamW, Embedding, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use scribe::semcor::{self, TaggedSentence};
use scribe::usage::model::UsageModel;
use scribe::usage::vocab::PAD;
use scribe::Lexicon;
use std::collections::HashMap;
use std::path::Path;

const WSD: &str = "data/semcor/wsdeval/WSD_Evaluation_Framework";
const WINDOW: usize = 40;
const STRIDE: usize = 20;
const MAX_RANK: usize = 8;
const MARGINS: &[f32] = &[0.0, 1.0, 2.0, 3.0, 4.0, 6.0];
const GLOSS_LEN: usize = 32;

fn load(name: &str, dir: &str) -> Vec<TaggedSentence> {
    let base = format!("{WSD}/{dir}/{name}");
    semcor::parse(&std::fs::read_to_string(format!("{base}.data.xml")).unwrap(), &std::fs::read_to_string(format!("{base}.gold.key.txt")).unwrap())
}

/// One target word: its candidate senses (dictionary order) and the gold ones.
struct Target {
    window: usize,
    piece: usize,
    senses: Vec<u32>,
    gold: Vec<u32>,
}

/// Windows (piece ids) and the targets read from them (each from the window where it's most central).
fn prepare(enc: &UsageModel, lex: &Lexicon, keys: &HashMap<String, u32>, sentences: &[TaggedSentence], train: bool)
    -> (Vec<Vec<u32>>, Vec<Target>, usize) {
    let (mut windows, mut targets, mut skipped) = (Vec::new(), Vec::new(), 0usize);
    for s in sentences {
        let words: Vec<String> = s.words.iter().map(|w| w.to_lowercase()).collect();
        let first = windows.len();
        let mut spans = Vec::new(); // (start, base pieces)
        let mut start = 0;
        loop {
            let end = (start + WINDOW).min(words.len());
            let (ids, base) = enc.encode_words(&words[start..end]);
            windows.push(ids);
            spans.push((start, base));
            if end == words.len() { break }
            start += STRIDE;
        }
        for inst in &s.instances {
            let senses = lex.senses(&inst.lemma, inst.pos).to_vec();
            let gold: Vec<u32> = inst.keys.iter().filter_map(|k| keys.get(k).copied()).filter(|g| senses.contains(g)).collect();
            if gold.is_empty() || senses.len() < 2 { if !train { skipped += 1 } continue }
            // The window where this word is most central and its base piece.
            let mut best: Option<(usize, usize, usize)> = None;
            for (w, (st, base)) in spans.iter().enumerate() {
                if inst.index < *st || inst.index - st >= base.len() { continue }
                let k = inst.index - st;
                let edge = k.min(base.len() - 1 - k);
                if best.is_none_or(|b| edge > b.0) { best = Some((edge, first + w, base[k])) }
            }
            match best {
                Some((_, w, piece)) => targets.push(Target { window: w, piece, senses, gold }),
                None => skipped += 1,
            }
        }
    }
    (windows, targets, skipped)
}

struct Head {
    synsets: Embedding,
    categories: Embedding,
    ranks: Embedding,
    rows: HashMap<u32, u32>,
    /// Piece ids of each synset's gloss, when --gloss is on.
    glosses: Option<Vec<Vec<u32>>>,
}

/// "members: definition" as encoder pieces, truncated to GLOSS_LEN.
fn gloss_pieces(enc: &UsageModel, lex: &Lexicon) -> Vec<Vec<u32>> {
    lex.synsets.iter().map(|s| {
        let text = format!("{} : {}", s.members.iter().take(3).cloned().collect::<Vec<_>>().join(" , "), s.definition);
        let words: Vec<String> = scribe::text::tokenize(&text).into_iter().map(|w| w.to_lowercase().replace('_', " ")).collect();
        let (ids, _) = enc.encode_words(&words);
        ids.into_iter().take(GLOSS_LEN).collect()
    }).collect()
}

/// Mean encoder vector over each listed synset's gloss: [u, dim].
fn gloss_vectors(enc: &UsageModel, glosses: &[Vec<u32>], synsets: &[u32], dev: &Device) -> candle_core::Result<Tensor> {
    let len = synsets.iter().map(|&s| glosses[s as usize].len()).max().unwrap_or(1).max(1);
    let (mut flat, mut mask) = (Vec::with_capacity(synsets.len() * len), Vec::with_capacity(synsets.len() * len));
    for &s in synsets {
        let g = &glosses[s as usize];
        flat.extend(g.iter().copied());
        flat.extend(std::iter::repeat_n(PAD, len - g.len()));
        mask.extend(std::iter::repeat_n(1f32, g.len()));
        mask.extend(std::iter::repeat_n(0f32, len - g.len()));
    }
    let u = synsets.len();
    let h = enc.hidden_states(&Tensor::from_vec(flat, (u, len), dev)?)?; // [u, len, dim]
    let mask = Tensor::from_vec(mask, (u, len, 1), dev)?;
    let count = mask.sum(1)?.clamp(1f32, f32::MAX)?;
    h.broadcast_mul(&mask)?.sum(1)?.broadcast_div(&count)
}

/// Candidate scores for a batch of targets: [n, K] (padding -1e9), K = most candidates.
fn score(enc: &UsageModel, head: &Head, lex: &Lexicon, windows: &[Vec<u32>], batch: &[&Target], dev: &Device)
    -> candle_core::Result<Tensor> {
    let mut used: Vec<usize> = batch.iter().map(|t| t.window).collect();
    used.sort();
    used.dedup();
    let wpos: HashMap<usize, usize> = used.iter().enumerate().map(|(i, &w)| (w, i)).collect();
    let len = used.iter().map(|&w| windows[w].len()).max().unwrap_or(1).max(1);
    let mut flat = Vec::with_capacity(used.len() * len);
    for &w in &used {
        flat.extend(windows[w].iter().copied());
        flat.extend(std::iter::repeat_n(PAD, len - windows[w].len()));
    }
    let hidden = enc.hidden_states(&Tensor::from_vec(flat, (used.len(), len), dev)?)?;
    let dim = hidden.dim(2)?;
    let hidden = hidden.reshape((used.len() * len, dim))?;
    let rows: Vec<u32> = batch.iter().map(|t| (wpos[&t.window] * len + t.piece) as u32).collect();
    let h = hidden.index_select(&Tensor::new(rows.as_slice(), dev)?, 0)?; // [n, dim]
    let k = batch.iter().map(|t| t.senses.len()).max().unwrap_or(1);
    let (mut srow, mut cat, mut rank, mut pad) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for t in batch {
        for j in 0..k {
            match t.senses.get(j) {
                Some(&s) => {
                    srow.push(head.rows.get(&s).copied().unwrap_or(0));
                    cat.push(lex.synsets[s as usize].category as u32);
                    rank.push(j.min(MAX_RANK - 1) as u32);
                    pad.push(0f32);
                }
                None => { srow.push(0); cat.push(0); rank.push(0); pad.push(-1e9) }
            }
        }
    }
    let n = batch.len();
    let mut sv = (head.synsets.forward(&Tensor::new(srow.as_slice(), dev)?)? + head.categories.forward(&Tensor::new(cat.as_slice(), dev)?)?)?;
    if let Some(glosses) = &head.glosses {
        // One gloss pass per distinct candidate synset in the batch.
        let cands: Vec<u32> = batch.iter().flat_map(|t| (0..k).map(move |j| t.senses.get(j).copied().unwrap_or(t.senses[0]))).collect();
        let mut distinct = cands.clone();
        distinct.sort();
        distinct.dedup();
        let at: HashMap<u32, u32> = distinct.iter().enumerate().map(|(i, &s)| (s, i as u32)).collect();
        let g = gloss_vectors(enc, glosses, &distinct, dev)?;
        let pick: Vec<u32> = cands.iter().map(|s| at[s]).collect();
        sv = (sv + g.index_select(&Tensor::new(pick.as_slice(), dev)?, 0)?)?;
    }
    let sv = sv.reshape((n, k, dim))?;
    let dot = sv.broadcast_mul(&h.reshape((n, 1, dim))?)?.sum(D::Minus1)?; // [n, k]
    let bias = head.ranks.forward(&Tensor::new(rank.as_slice(), dev)?)?.reshape((n, k))?;
    (dot + bias)? + Tensor::from_vec(pad, (n, k), dev)?
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage_name = args.get(1).cloned().unwrap_or_else(|| "usage".into());
    let epochs: usize = args.get(2).and_then(|e| e.parse().ok()).unwrap_or(3);
    let use_gloss = args.iter().any(|a| a == "--gloss");
    let enc_lr: f64 = args.iter().position(|a| a == "--enc-lr").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(5e-5);
    #[cfg(feature = "metal")]
    let dev = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let dev = Device::Cpu;

    let lex = Lexicon::load(Path::new("models/lexicon.bin"))?;
    let keys = semcor::sense_keys(Path::new("data/oewn"), &lex)?;
    let train = load("semcor", "Training_Corpora/SemCor");
    let all = load("ALL", "Evaluation_Datasets/ALL");
    let enc = UsageModel::load(&Path::new("models").join(&usage_name), dev.clone())?;
    let (tr_w, tr_t, _) = prepare(&enc, &lex, &keys, &train, true);
    let (all_w, all_t, skipped) = prepare(&enc, &lex, &keys, &all, false);
    let first = all_t.iter().filter(|t| t.gold.contains(&t.senses[0])).count();
    println!("{} training targets; ALL: {} ambiguous targets ({skipped} skipped), first-listed {:.1}%",
             tr_t.len(), all_t.len(), 100.0 * first as f64 / all_t.len() as f64);

    // One synset row per gold sense seen in training (row 0 = every other synset).
    let mut rows = HashMap::new();
    for t in &tr_t {
        let next = rows.len() as u32 + 1;
        rows.entry(t.gold[0]).or_insert(next);
    }
    let dim = enc.config.dim;
    let map = VarMap::new();
    let vb = VarBuilder::from_varmap(&map, DType::F32, &dev);
    let head = Head {
        synsets: embedding(rows.len() + 1, dim, vb.pp("synsets"))?,
        categories: embedding(lex.categories.len(), dim, vb.pp("categories"))?,
        ranks: embedding(MAX_RANK, 1, vb.pp("ranks"))?,
        rows,
        glosses: use_gloss.then(|| gloss_pieces(&enc, &lex)),
    };
    println!("gloss vectors: {}", if use_gloss { "on" } else { "off" });
    let mut enc_opt = AdamW::new(enc.trainable_vars(), ParamsAdamW { lr: enc_lr, weight_decay: 0.01, ..Default::default() })?;
    let mut head_opt = AdamW::new(map.all_vars(), ParamsAdamW { lr: 2e-3, weight_decay: 0.01, ..Default::default() })?;

    let evaluate = || -> candle_core::Result<Vec<f64>> {
        let mut right = vec![0usize; MARGINS.len()];
        let refs: Vec<&Target> = all_t.iter().collect();
        for chunk in refs.chunks(128) {
            let s = score(&enc, &head, &lex, &all_w, chunk, &dev)?.to_vec2::<f32>()?;
            for (t, row) in chunk.iter().zip(s) {
                let n = t.senses.len();
                let best = (0..n).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap_or(0);
                for (mi, &m) in MARGINS.iter().enumerate() {
                    let pick = if row[best] - row[0] > m { best } else { 0 };
                    right[mi] += t.gold.contains(&t.senses[pick]) as usize;
                }
            }
        }
        Ok(right.into_iter().map(|r| 100.0 * r as f64 / all_t.len() as f64).collect())
    };

    let mut order: Vec<usize> = (0..tr_t.len()).collect();
    let mut rng = 0x2545F4914F6CDD1Du64;
    for epoch in 0..epochs {
        for i in (1..order.len()).rev() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        // Batches of targets grouped by window so a window is encoded once per batch.
        order.sort_by_key(|&i| tr_t[i].window / 64);
        let mut batches: Vec<Vec<usize>> = order.chunks(64).map(|c| c.to_vec()).collect();
        for i in (1..batches.len()).rev() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            batches.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        let t0 = std::time::Instant::now();
        for b in &batches {
            let batch: Vec<&Target> = b.iter().map(|&i| &tr_t[i]).collect();
            let logits = score(&enc, &head, &lex, &tr_w, &batch, &dev)?;
            let gold: Vec<u32> = batch.iter().map(|t| t.senses.iter().position(|s| *s == t.gold[0]).unwrap() as u32).collect();
            let loss = candle_nn::loss::cross_entropy(&logits, &Tensor::new(gold.as_slice(), &dev)?)?;
            let grads = loss.backward()?;
            enc_opt.step(&grads)?;
            head_opt.step(&grads)?;
        }
        let scores = evaluate()?;
        let cells: Vec<String> = MARGINS.iter().zip(&scores).map(|(m, s)| format!("m{m}: {s:.1}%")).collect();
        println!("epoch {}: ALL {}  ({:.0?})", epoch + 1, cells.join("  "), t0.elapsed());
    }
    Ok(())
}
