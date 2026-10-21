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
//!
//! --stack: also give it the shipped parser's choices (jackknifed on train):
//! a learned bonus on each word's prior head, and the prior relation as a
//! label input. Batches group sentences of similar length.
//!
//! Each epoch prints dev and test; the best epoch by dev is saved to
//! models/syntax-<usage>[-stacked]/ (usage model + parser_head.safetensors/.json).
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
    /// Learned weight on "the shipped parser chose this head".
    prior_bonus: Tensor,
    /// UPOS tagging head (multi-task: tagging and parsing share the encoder).
    tag_hidden: Linear,
    tag_out: Linear,
}

/// The shipped parser's choice per word: (head, relation index).
type Prior = Vec<(usize, u32)>;

impl Model {
    fn new(vb: VarBuilder, input: usize, labels: usize, tags: usize, stack: bool) -> candle_core::Result<Model> {
        Ok(Model {
            chars: embedding(CHAR_BUCKETS, CHAR_DIM, vb.pp("chars"))?,
            root: vb.get_with_hints((1, input), "root", candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 })?,
            arc_dep: linear(input, ARC, vb.pp("arc_dep"))?,
            arc_head: linear(input, ARC, vb.pp("arc_head"))?,
            arc_u: vb.get_with_hints((ARC, ARC), "arc_u", candle_nn::Init::Const(0.0))?,
            arc_bias: linear(ARC, 1, vb.pp("arc_bias"))?,
            lab_dep: linear(input, LAB, vb.pp("lab_dep"))?,
            lab_head: linear(input, LAB, vb.pp("lab_head"))?,
            lab_out: linear(3 * LAB + if stack { labels } else { 0 }, labels, vb.pp("lab_out"))?,
            prior_bonus: vb.get_with_hints(1, "prior_bonus", candle_nn::Init::Const(if stack { 1.0 } else { 0.0 }))?,
            tag_hidden: linear(input, 256, vb.pp("tag_hidden"))?,
            tag_out: linear(256, tags, vb.pp("tag_out"))?,
        })
    }
}

fn tag_logits(m: &Model, xs: &[Tensor]) -> candle_core::Result<Tensor> {
    let x = Tensor::cat(xs, 0)?;
    m.tag_out.forward(&m.tag_hidden.forward(&x)?.relu()?)
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

/// Batched scoring. `xs`: word matrices [n_i, input]. Returns arc scores
/// [b, L, L+1] (dependent × head, head 0 = root; padded heads masked to -1e9)
/// and the label projections [b, L, LAB] (dependents) / [b, L+1, LAB] (heads).
fn scores(m: &Model, xs: &[Tensor], priors: Option<&[&Prior]>, dev: &Device) -> candle_core::Result<(Tensor, Tensor, Tensor, usize)> {
    let b = xs.len();
    let l = xs.iter().map(|x| x.dim(0).unwrap()).max().unwrap_or(1);
    let input = xs[0].dim(1)?;
    let mut padded = Vec::with_capacity(b);
    let mut mask = vec![0f32; b * (l + 1)];
    for (i, x) in xs.iter().enumerate() {
        let n = x.dim(0)?;
        let rows = Tensor::cat(&[m.root.clone(), x.clone()], 0)?;
        let rows = if n < l { Tensor::cat(&[rows, Tensor::zeros((l - n, input), DType::F32, dev)?], 0)? } else { rows };
        padded.push(rows);
        for j in n + 1..=l { mask[i * (l + 1) + j] = -1e9 }
    }
    let with_root = Tensor::stack(&padded, 0)?; // [b, L+1, input]
    let words = with_root.narrow(1, 1, l)?; // [b, L, input]
    let dep = m.arc_dep.forward(&words)?.relu()?; // [b, L, ARC]
    let head = m.arc_head.forward(&with_root)?.relu()?; // [b, L+1, ARC]
    let bil = dep.broadcast_matmul(&m.arc_u)?.matmul(&head.transpose(1, 2)?.contiguous()?)?; // [b, L, L+1]
    let bias = m.arc_bias.forward(&head)?.transpose(1, 2)?; // [b, 1, L+1]
    let mut arcs = bil.broadcast_add(&bias)?.broadcast_add(&Tensor::from_vec(mask, (b, 1, l + 1), dev)?)?;
    if let Some(priors) = priors {
        let mut ind = vec![0f32; b * l * (l + 1)];
        for (si, p) in priors.iter().enumerate() {
            for (k, &(h, _)) in p.iter().enumerate() { ind[si * l * (l + 1) + k * (l + 1) + h] = 1.0 }
        }
        let ind = Tensor::from_vec(ind, (b, l, l + 1), dev)?;
        arcs = (arcs + ind.broadcast_mul(&m.prior_bonus.reshape((1, 1, 1))?)?)?;
    }
    let ld = m.lab_dep.forward(&words)?.relu()?;
    let lh = m.lab_head.forward(&with_root)?.relu()?;
    Ok((arcs, ld, lh, l))
}

/// Label logits for (sentence, dependent, head) triples.
fn label_logits(m: &Model, ld: &Tensor, lh: &Tensor, triples: &[(usize, usize, usize)], l: usize, prior_labels: Option<(&[u32], usize)>, dev: &Device) -> candle_core::Result<Tensor> {
    let lab = ld.dim(2)?;
    let ld = ld.reshape((ld.dim(0)? * l, lab))?;
    let lh = lh.reshape((lh.dim(0)? * (l + 1), lab))?;
    let di: Vec<u32> = triples.iter().map(|&(s, d, _)| (s * l + d) as u32).collect();
    let hi: Vec<u32> = triples.iter().map(|&(s, _, h)| (s * (l + 1) + h) as u32).collect();
    let d = ld.index_select(&Tensor::new(di.as_slice(), dev)?, 0)?;
    let h = lh.index_select(&Tensor::new(hi.as_slice(), dev)?, 0)?;
    let mut parts = vec![d.clone(), h.clone(), (d * &h)?];
    if let Some((pl, n)) = prior_labels {
        let mut one = vec![0f32; pl.len() * n];
        for (k, &l) in pl.iter().enumerate() { one[k * n + l as usize] = 1.0 }
        parts.push(Tensor::from_vec(one, (pl.len(), n), dev)?);
    }
    m.lab_out.forward(&Tensor::cat(&parts, 1)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage_name = args.get(1).cloned().unwrap_or_else(|| "usage-morph".into());
    // Tagging + parsing in one model: saved as models/syntax-<usage>[-stacked]/.
    let epochs: usize = args.get(2).and_then(|e| e.parse().ok()).unwrap_or(5);
    let stack = args.iter().any(|a| a == "--stack");
    let enc_lr: f64 = args.iter().position(|a| a == "--enc-lr").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(5e-5);
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

    // Shipped-parser choices: shipped models on dev/test; jackknifed on train
    // (parsers trained on 4/5 with jackknifed tags, parsing the held-out fifth).
    let tag_strings = |g: &Vec<usize>| -> Vec<String> { g.iter().map(|&i| upos[i].clone()).collect() };
    let parse_with = |p: &Parser, s: &Sentence, g: &Vec<usize>| -> Prior {
        let words: Vec<&str> = s.iter().map(|t| t.form.as_str()).collect();
        let tags = tag_strings(g);
        let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
        p.parse(&words, &refs).into_iter().map(|a| (a.head.min(s.len()), li(&a.label))).collect()
    };
    let (dev_prior, test_prior, train_prior): (Vec<Prior>, Vec<Prior>, Vec<Prior>) = if stack {
        let dp = devset.iter().zip(&dev_g).map(|(s, g)| parse_with(&shipped_parser, s, g)).collect();
        let tp = test.iter().zip(&test_g).map(|(s, g)| parse_with(&shipped_parser, s, g)).collect();
        let mut trp: Vec<Prior> = vec![Vec::new(); train.len()];
        for fold in 0..5 {
            let rest: Vec<usize> = (0..train.len()).filter(|i| i % 5 != fold).collect();
            let sents: Vec<Sentence> = rest.iter().map(|&i| train[i].clone()).collect();
            let tags: Vec<Vec<String>> = rest.iter().map(|&i| tag_strings(&train_g[i])).collect();
            let p = Parser::train(&sents, &tags, 12);
            for i in (0..train.len()).filter(|i| i % 5 == fold) { trp[i] = parse_with(&p, &train[i], &train_g[i]) }
            eprintln!("  jackknifed parser fold {} done", fold + 1);
        }
        (dp, tp, trp)
    } else {
        (vec![Vec::new(); devset.len()], vec![Vec::new(); test.len()], vec![Vec::new(); train.len()])
    };

    let enc = UsageModel::load(&Path::new("models").join(&usage_name), dev.clone())?;
    let prep = |set: &[Sentence]| set.iter().map(|s| prepare(&enc, s)).collect::<Vec<_>>();
    let (tr_p, dev_p, te_p) = (prep(&train), prep(&devset), prep(&test));
    let input = enc.config.dim + CHAR_DIM + upos.len();
    let heads_map = VarMap::new();
    let m = Model::new(VarBuilder::from_varmap(&heads_map, DType::F32, &dev), input, labels.len(), upos.len(), stack)?;
    let mut enc_opt = AdamW::new(enc.trainable_vars(), ParamsAdamW { lr: enc_lr, weight_decay: 0.01, ..Default::default() })?;
    let mut head_opt = AdamW::new(heads_map.all_vars(), ParamsAdamW { lr: 1e-3, weight_decay: 1e-4, ..Default::default() })?;

    let evaluate = |set: &[Sentence], p: &[Prepared], g: &[Vec<usize>], prior: &[Prior]| -> candle_core::Result<(f64, f64, f64)> {
        let (mut uas, mut las, mut n, mut tags_right) = (0usize, 0usize, 0usize, 0usize);
        let idx: Vec<usize> = (0..set.len()).collect();
        for chunk in idx.chunks(32) {
            let batch: Vec<_> = chunk.iter().map(|&i| (&set[i], &p[i], &g[i])).collect();
            let xs = word_vectors(&enc, &m, &batch, upos.len(), &dev)?;
            let tag_pred = tag_logits(&m, &xs)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
            let gold_tags: Vec<u32> = batch.iter().flat_map(|(s, _, _)| s.iter().map(|t| ti(&t.upos) as u32)).collect();
            tags_right += tag_pred.iter().zip(&gold_tags).filter(|(a, b)| a == b).count();
            let pr: Vec<&Prior> = chunk.iter().map(|&i| &prior[i]).collect();
            let (arcs, ld, lh, l) = scores(&m, &xs, stack.then_some(&pr[..]), &dev)?;
            let heads = arcs.argmax(D::Minus1)?.to_vec2::<u32>()?;
            let plabels: Vec<u32> = if stack { chunk.iter().flat_map(|&i| prior[i].iter().map(|&(_, l)| l)).collect() } else { Vec::new() };
            let mut triples = Vec::new();
            for (si, (s, _, _)) in batch.iter().enumerate() {
                for k in 0..s.len() { triples.push((si, k, heads[si][k] as usize)) }
            }
            let labs = label_logits(&m, &ld, &lh, &triples, l, stack.then_some((&plabels[..], labels.len())), &dev)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
            let mut t_i = 0;
            for (si, (s, _, _)) in batch.iter().enumerate() {
                for (k, t) in s.iter().enumerate() {
                    if heads[si][k] as usize == t.head {
                        uas += 1;
                        las += (labs[t_i] == li(&t.deprel)) as usize;
                    }
                    t_i += 1;
                    n += 1;
                }
            }
        }
        Ok((100.0 * uas as f64 / n as f64, 100.0 * las as f64 / n as f64, 100.0 * tags_right as f64 / n as f64))
    };

    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut rng = 0x2545F4914F6CDD1Du64;
    let mut next = move |n: usize| { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; (rng % n as u64) as usize };
    let mut best = f64::MIN;
    for epoch in 0..epochs {
        for i in (1..order.len()).rev() { order.swap(i, next(i + 1)) }
        // Length buckets: sort within pools of 50 batches, then shuffle the batches.
        let mut batches: Vec<Vec<usize>> = Vec::new();
        for pool in order.chunks(32 * 50) {
            let mut pool = pool.to_vec();
            pool.sort_by_key(|&i| train[i].len());
            batches.extend(pool.chunks(32).map(|c| c.to_vec()));
        }
        for i in (1..batches.len()).rev() { batches.swap(i, next(i + 1)) }
        let t0 = std::time::Instant::now();
        for chunk in &batches {
            let batch: Vec<_> = chunk.iter().map(|&i| (&train[i], &tr_p[i], &train_g[i])).collect();
            let xs = word_vectors(&enc, &m, &batch, upos.len(), &dev)?;
            let pr: Vec<&Prior> = chunk.iter().map(|&i| &train_prior[i]).collect();
            let (arcs, ld, lh, l) = scores(&m, &xs, stack.then_some(&pr[..]), &dev)?;
            let plabels: Vec<u32> = if stack { chunk.iter().flat_map(|&i| train_prior[i].iter().map(|&(_, l)| l)).collect() } else { Vec::new() };
            // Real words only: rows of the flattened [b*L, L+1] arc scores.
            let (mut rows, mut gold_heads, mut triples, mut gold_labels) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (si, (s, _, _)) in batch.iter().enumerate() {
                for (k, t) in s.iter().enumerate() {
                    rows.push((si * l + k) as u32);
                    gold_heads.push(t.head as u32);
                    triples.push((si, k, t.head));
                    gold_labels.push(li(&t.deprel));
                }
            }
            let flat = arcs.reshape((batch.len() * l, l + 1))?.index_select(&Tensor::new(rows.as_slice(), &dev)?, 0)?;
            let arc_loss = candle_nn::loss::cross_entropy(&flat, &Tensor::new(gold_heads.as_slice(), &dev)?)?;
            let lab_loss = candle_nn::loss::cross_entropy(&label_logits(&m, &ld, &lh, &triples, l, stack.then_some((&plabels[..], labels.len())), &dev)?, &Tensor::new(gold_labels.as_slice(), &dev)?)?;
            let gold_tags: Vec<u32> = batch.iter().flat_map(|(s, _, _)| s.iter().map(|t| ti(&t.upos) as u32)).collect();
            let tag_loss = candle_nn::loss::cross_entropy(&tag_logits(&m, &xs)?, &Tensor::new(gold_tags.as_slice(), &dev)?)?;
            let loss = ((arc_loss + lab_loss)? + tag_loss)?;
            let grads = loss.backward()?;
            enc_opt.step(&grads)?;
            head_opt.step(&grads)?;
        }
        let train_time = t0.elapsed();
        let (u, l, tg) = evaluate(&devset, &dev_p, &dev_g, &dev_prior)?;
        let (tu, tl, ttg) = evaluate(&test, &te_p, &test_g, &test_prior)?;
        let better = u + l + tg > best;
        println!("epoch {}: dev UAS {u:.2}% LAS {l:.2}% UPOS {tg:.2}%  test UAS {tu:.2}% LAS {tl:.2}% UPOS {ttg:.2}%  (train {train_time:.0?}, total {:.0?}){}",
                 epoch + 1, t0.elapsed(), if better { "  saved" } else { "" });
        if better {
            // The best epoch by dev score: fine-tuned usage model + parser head + label/tag lists.
            best = u + l + tg;
            let out = Path::new("models").join(format!("syntax-{usage_name}{}", if stack { "-stacked" } else { "" }));
            enc.save(&out)?;
            heads_map.save(out.join("parser_head.safetensors"))?;
            let meta = serde_json::json!({ "labels": labels, "upos": upos, "stack": stack, "epoch": epoch + 1,
                "dev": { "uas": u, "las": l, "upos": tg }, "test": { "uas": tu, "las": tl, "upos": ttg } });
            std::fs::write(out.join("parser_head.json"), serde_json::to_vec_pretty(&meta)?)?;
        }
    }
    Ok(())
}
