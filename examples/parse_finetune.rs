//! Dependency parsing with a biaffine head (Dozat & Manning) on the fine-tuned
//! usage model, trained on UD English EWT and scored on dev/test against the
//! shipped transition parser (both with predicted tags).
//!
//!   cargo run --release --features metal --example parse_finetune [usage-dir] [epochs]
//!
//! Word representation: the usage model's vector (sentences longer than its
//! window are stitched from overlapping windows), hashed spelling features,
//! and the perceptron's UPOS tag (jackknifed on train). Arcs: biaffine scores
//! of every word as head of every other (argmax head per word). Labels: an MLP
//! over the dependent and its head. LAS counts base relations (before ':'),
//! as the shipped parser's evaluation does. Nothing is saved.
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, linear, AdamW, Embedding, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use scribe::conllu::{self, Sentence};
use scribe::parser::Parser;
use scribe::tagger::{Column, Tagger};
use scribe::usage::model::UsageModel;
use scribe::usage::vocab::PAD;
use scribe::Lexicon;
use std::path::Path;

const WINDOW: usize = 40;
const STRIDE: usize = 20;
const CHAR_BUCKETS: usize = 1 << 16;
const CHAR_DIM: usize = 64;
const ARC: usize = 256;
const LAB: usize = 128;

fn base(rel: &str) -> &str {
    rel.split(':').next().unwrap_or(rel)
}

fn spelling(word: &str) -> Vec<u32> {
    let chars: Vec<char> = word.chars().collect();
    let lower: Vec<char> = word.to_lowercase().chars().collect();
    let mut feats: Vec<String> = Vec::new();
    for k in 1..=4.min(lower.len()) { feats.push(format!("s{k}:{}", lower[lower.len() - k..].iter().collect::<String>())) }
    for k in 1..=3.min(lower.len()) { feats.push(format!("p{k}:{}", lower[..k].iter().collect::<String>())) }
    let mut shape = String::new();
    for c in chars.iter().map(|c| if c.is_uppercase() { 'X' } else if c.is_lowercase() { 'x' } else if c.is_ascii_digit() { 'd' } else { *c }) {
        if !shape.ends_with(c) { shape.push(c) }
    }
    feats.push(format!("shape:{shape}"));
    feats.push(format!("len:{}", chars.len().min(10)));
    feats.iter().map(|f| {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in f.bytes() { h = (h ^ b as u64).wrapping_mul(0x100000001b3) }
        (h % CHAR_BUCKETS as u64) as u32
    }).collect()
}

/// A sentence's windows: piece ids per window, and for each word the
/// (window, base piece) it reads its vector from — the window where it's most central.
struct Prepared {
    windows: Vec<Vec<u32>>,
    source: Vec<(usize, usize)>,
}

fn prepare(model: &UsageModel, s: &Sentence) -> Prepared {
    let words: Vec<String> = s.iter().map(|t| t.form.to_lowercase()).collect();
    let mut windows = Vec::new();
    let mut source: Vec<(usize, usize, usize)> = vec![(usize::MAX, 0, 0); words.len()]; // (edge rank, window, piece)
    let mut start = 0;
    loop {
        let end = (start + WINDOW).min(words.len());
        let (ids, basep) = model.encode_words(&words[start..end]);
        let w = windows.len();
        let n = basep.len();
        for (k, piece) in basep.into_iter().enumerate() {
            let edge = k.min(n - 1 - k);
            let slot = &mut source[start + k];
            if slot.0 == usize::MAX || edge > slot.0 { *slot = (edge, w, piece) }
        }
        windows.push(ids);
        if end == words.len() { break }
        start += STRIDE;
    }
    // Words past every window's piece limit (very long tokens) fall back to the first window's first piece.
    Prepared { windows, source: source.into_iter().map(|(e, w, p)| if e == usize::MAX { (0, 0) } else { (w, p) }).collect() }
}

struct Model {
    chars: Embedding,
    root: Tensor,
    arc_dep: Linear,
    arc_head: Linear,
    arc_u: Tensor,
    arc_bias: Linear,
    lab_dep: Linear,
    lab_head: Linear,
    lab_out: Linear,
}

impl Model {
    fn new(vb: VarBuilder, input: usize, labels: usize) -> candle_core::Result<Model> {
        Ok(Model {
            chars: embedding(CHAR_BUCKETS, CHAR_DIM, vb.pp("chars"))?,
            root: vb.get_with_hints((1, input), "root", candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 })?,
            arc_dep: linear(input, ARC, vb.pp("arc_dep"))?,
            arc_head: linear(input, ARC, vb.pp("arc_head"))?,
            arc_u: vb.get_with_hints((ARC, ARC), "arc_u", candle_nn::Init::Const(0.0))?,
            arc_bias: linear(ARC, 1, vb.pp("arc_bias"))?,
            lab_dep: linear(input, LAB, vb.pp("lab_dep"))?,
            lab_head: linear(input, LAB, vb.pp("lab_head"))?,
            lab_out: linear(3 * LAB, labels, vb.pp("lab_out"))?,
        })
    }
}

/// Word matrices [n, input] for a batch of sentences, from one encoder pass over all their windows.
fn word_vectors(enc: &UsageModel, m: &Model, batch: &[(&Sentence, &Prepared, &Vec<usize>)], tags: usize, dev: &Device)
    -> candle_core::Result<Vec<Tensor>> {
    let all: Vec<&Vec<u32>> = batch.iter().flat_map(|(_, p, _)| p.windows.iter()).collect();
    let len = all.iter().map(|w| w.len()).max().unwrap_or(1).max(1);
    let mut flat = Vec::with_capacity(all.len() * len);
    for w in &all {
        flat.extend(w.iter().copied());
        flat.extend(std::iter::repeat_n(PAD, len - w.len()));
    }
    let hidden = enc.hidden_states(&Tensor::from_vec(flat, (all.len(), len), dev)?)?;
    let dim = hidden.dim(2)?;
    let hidden = hidden.reshape((all.len() * len, dim))?;
    let mut out = Vec::new();
    let mut window_offset = 0;
    for (s, p, g) in batch {
        let rows: Vec<u32> = p.source.iter().map(|&(w, piece)| ((window_offset + w) * len + piece) as u32).collect();
        window_offset += p.windows.len();
        let h = hidden.index_select(&Tensor::new(rows.as_slice(), dev)?, 0)?;
        // Spelling: mean of hashed feature embeddings.
        let (mut ids, mut owner) = (Vec::new(), Vec::new());
        for (k, t) in s.iter().enumerate() {
            for f in spelling(&t.form) { ids.push(f); owner.push(k as u32) }
        }
        let n = s.len();
        let mut counts = vec![0f32; n];
        for &o in &owner { counts[o as usize] += 1.0 }
        let e = m.chars.forward(&Tensor::new(ids.as_slice(), dev)?)?;
        let sp = Tensor::zeros((n, CHAR_DIM), DType::F32, dev)?.index_add(&Tensor::new(owner.as_slice(), dev)?, &e, 0)?
            .broadcast_div(&Tensor::from_vec(counts, (n, 1), dev)?)?;
        let mut one = vec![0f32; n * tags];
        for (k, &t) in g.iter().enumerate() { one[k * tags + t] = 1.0 }
        out.push(Tensor::cat(&[h, sp, Tensor::from_vec(one, (n, tags), dev)?], 1)?);
    }
    Ok(out)
}

/// Arc scores [n, n+1] (dependent × head, head 0 = root) and the head/dep label projections.
fn scores(m: &Model, x: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    let with_root = Tensor::cat(&[m.root.clone(), x.clone()], 0)?; // [n+1, input]
    let dep = m.arc_dep.forward(x)?.relu()?; // [n, ARC]
    let head = m.arc_head.forward(&with_root)?.relu()?; // [n+1, ARC]
    let bil = dep.matmul(&m.arc_u)?.matmul(&head.t()?)?; // [n, n+1]
    let bias = m.arc_bias.forward(&head)?.t()?; // [1, n+1]
    let arcs = bil.broadcast_add(&bias)?;
    let ld = m.lab_dep.forward(x)?.relu()?; // [n, LAB]
    let lh = m.lab_head.forward(&with_root)?.relu()?; // [n+1, LAB]
    Ok((arcs, ld, lh))
}

fn label_logits(m: &Model, ld: &Tensor, lh: &Tensor, heads: &[u32], dev: &Device) -> candle_core::Result<Tensor> {
    let h = lh.index_select(&Tensor::new(heads, dev)?, 0)?;
    m.lab_out.forward(&Tensor::cat(&[ld.clone(), h.clone(), (ld * &h)?], 1)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage_name = args.get(1).cloned().unwrap_or_else(|| "usage-morph".into());
    let epochs: usize = args.get(2).and_then(|e| e.parse().ok()).unwrap_or(5);
    #[cfg(feature = "metal")]
    let dev = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let dev = Device::Cpu;

    let load = |s: &str| conllu::parse(&std::fs::read_to_string(format!("data/en_ewt-ud-{s}.conllu")).unwrap());
    let (train, devset, test) = (load("train"), load("dev"), load("test"));
    let lex = Lexicon::load(Path::new("models/lexicon.bin"))?;

    // Shipped baseline: transition parser with predicted tags.
    let shipped_tagger = Tagger::load(Path::new("models/tagger.bin"))?;
    let shipped_parser = Parser::load(Path::new("models/parser.bin"))?;
    let tag = |w: &[&str]| shipped_tagger.tag(w, Some(&lex)).into_iter().map(String::from).collect::<Vec<_>>();
    let (bu, bl) = shipped_parser.evaluate(&devset, &tag);
    let (tu, tl) = shipped_parser.evaluate(&test, &tag);
    println!("shipped parser: dev UAS {bu:.2}% LAS {bl:.2}%  test UAS {tu:.2}% LAS {tl:.2}%");

    // UPOS guesses: shipped tagger on dev/test, jackknifed on train.
    let mut upos: Vec<String> = train.iter().flatten().map(|t| t.upos.clone()).collect();
    upos.sort();
    upos.dedup();
    let ti = |t: &str| upos.iter().position(|x| x == t).unwrap_or(0);
    let guess = |t: &Tagger, set: &[Sentence]| -> Vec<Vec<usize>> {
        set.iter().map(|s| {
            let w: Vec<&str> = s.iter().map(|t| t.form.as_str()).collect();
            t.tag(&w, Some(&lex)).into_iter().map(ti).collect()
        }).collect()
    };
    let (dev_g, test_g) = (guess(&shipped_tagger, &devset), guess(&shipped_tagger, &test));
    let mut train_g = vec![Vec::new(); train.len()];
    for fold in 0..5 {
        let rest: Vec<Sentence> = train.iter().enumerate().filter(|(i, _)| i % 5 != fold).map(|(_, s)| s.clone()).collect();
        let held: Vec<usize> = (0..train.len()).filter(|i| i % 5 == fold).collect();
        let t = Tagger::train(&rest, Column::Upos, Some(&lex), 5);
        let sents: Vec<Sentence> = held.iter().map(|&i| train[i].clone()).collect();
        for (&i, g) in held.iter().zip(guess(&t, &sents)) { train_g[i] = g }
    }

    let mut labels: Vec<String> = train.iter().flatten().map(|t| base(&t.deprel).to_string()).collect();
    labels.sort();
    labels.dedup();
    let li = |l: &str| labels.iter().position(|x| x == base(l)).unwrap_or(0) as u32;

    let enc = UsageModel::load(&Path::new("models").join(&usage_name), dev.clone())?;
    let prep = |set: &[Sentence]| set.iter().map(|s| prepare(&enc, s)).collect::<Vec<_>>();
    let (tr_p, dev_p, te_p) = (prep(&train), prep(&devset), prep(&test));
    let input = enc.config.dim + CHAR_DIM + upos.len();
    let heads_map = VarMap::new();
    let m = Model::new(VarBuilder::from_varmap(&heads_map, DType::F32, &dev), input, labels.len())?;
    let mut enc_opt = AdamW::new(enc.trainable_vars(), ParamsAdamW { lr: 5e-5, weight_decay: 0.01, ..Default::default() })?;
    let mut head_opt = AdamW::new(heads_map.all_vars(), ParamsAdamW { lr: 1e-3, weight_decay: 1e-4, ..Default::default() })?;

    let evaluate = |set: &[Sentence], p: &[Prepared], g: &[Vec<usize>]| -> candle_core::Result<(f64, f64)> {
        let (mut uas, mut las, mut n) = (0usize, 0usize, 0usize);
        let idx: Vec<usize> = (0..set.len()).collect();
        for chunk in idx.chunks(32) {
            let batch: Vec<_> = chunk.iter().map(|&i| (&set[i], &p[i], &g[i])).collect();
            for ((s, _, _), x) in batch.iter().zip(word_vectors(&enc, &m, &batch, upos.len(), &dev)?) {
                let (arcs, ld, lh) = scores(&m, &x)?;
                let heads = arcs.argmax(D::Minus1)?.to_vec1::<u32>()?;
                let labs = label_logits(&m, &ld, &lh, &heads, &dev)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
                for (k, t) in s.iter().enumerate() {
                    if heads[k] as usize == t.head {
                        uas += 1;
                        las += (labs[k] == li(&t.deprel)) as usize;
                    }
                    n += 1;
                }
            }
        }
        Ok((100.0 * uas as f64 / n as f64, 100.0 * las as f64 / n as f64))
    };

    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut rng = 0x2545F4914F6CDD1Du64;
    for epoch in 0..epochs {
        for i in (1..order.len()).rev() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        let t0 = std::time::Instant::now();
        for chunk in order.chunks(16) {
            let batch: Vec<_> = chunk.iter().map(|&i| (&train[i], &tr_p[i], &train_g[i])).collect();
            let xs = word_vectors(&enc, &m, &batch, upos.len(), &dev)?;
            let mut losses = Vec::new();
            for ((s, _, _), x) in batch.iter().zip(xs) {
                let (arcs, ld, lh) = scores(&m, &x)?;
                let gold_heads: Vec<u32> = s.iter().map(|t| t.head as u32).collect();
                let gold_labels: Vec<u32> = s.iter().map(|t| li(&t.deprel)).collect();
                let arc_loss = candle_nn::loss::cross_entropy(&arcs, &Tensor::new(gold_heads.as_slice(), &dev)?)?;
                let lab_loss = candle_nn::loss::cross_entropy(&label_logits(&m, &ld, &lh, &gold_heads, &dev)?, &Tensor::new(gold_labels.as_slice(), &dev)?)?;
                losses.push((arc_loss + lab_loss)?);
            }
            let loss = (Tensor::stack(&losses, 0)?.sum_all()? / losses.len() as f64)?;
            let grads = loss.backward()?;
            enc_opt.step(&grads)?;
            head_opt.step(&grads)?;
        }
        let (u, l) = evaluate(&devset, &dev_p, &dev_g)?;
        println!("epoch {}: dev UAS {u:.2}% LAS {l:.2}% ({:.0?})", epoch + 1, t0.elapsed());
    }
    let (u, l) = evaluate(&test, &te_p, &test_g)?;
    println!("biaffine ({usage_name}): test UAS {u:.2}% LAS {l:.2}%");
    Ok(())
}
