//! Part-of-speech tagging with a small head on the usage model's contextual
//! vectors, against the shipped averaged-perceptron tagger.
//!
//!   cargo run --release --example tag_head_eval [usage-dir] [upos|xpos]
//!
//! Heads (trained on UD English EWT train, scored on dev and test):
//!   linear:      vector → tag
//!   mlp:         vector → 256 → tag
//!   mlp+tagger:  [vector, perceptron's tag one-hot] → 256 → tag; on train the
//!                perceptron tags are jackknifed (5 folds) so the head doesn't
//!                learn to trust tags that are always right on train.
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use scribe::conllu::{self, Sentence};
use scribe::tagger::{Column, Tagger};
use scribe::usage::model::UsageModel;
use scribe::Lexicon;
use std::path::Path;

const WINDOW: usize = 40;
const STRIDE: usize = 20;

/// Contextual vector per word; sentences longer than the model's window go
/// through overlapping windows, each word taking the window where it's most central.
fn vectors(model: &UsageModel, sentences: &[Sentence]) -> Vec<Vec<Vec<f32>>> {
    let dim = model.config.dim;
    let mut jobs: Vec<(usize, usize, Vec<String>)> = Vec::new(); // (sentence, start, words)
    for (si, s) in sentences.iter().enumerate() {
        let words: Vec<String> = s.iter().map(|t| t.form.to_lowercase()).collect();
        let mut start = 0;
        loop {
            let end = (start + WINDOW).min(words.len());
            jobs.push((si, start, words[start..end].to_vec()));
            if end == words.len() { break }
            start += STRIDE;
        }
    }
    let mut out: Vec<Vec<Vec<f32>>> = sentences.iter().map(|s| vec![vec![0.0; dim]; s.len()]).collect();
    let mut best: Vec<Vec<usize>> = sentences.iter().map(|s| vec![usize::MAX; s.len()]).collect();
    for (n, chunk) in jobs.chunks(256).enumerate() {
        if n % 20 == 0 { eprintln!("  vectors {}/{}", n * 256, jobs.len()) }
        let batch: Vec<Vec<String>> = chunk.iter().map(|(_, _, w)| w.clone()).collect();
        let hidden = model.contextual_batch(&batch).expect("contextual");
        for ((si, start, words), h) in chunk.iter().zip(hidden) {
            for (k, v) in h.into_iter().enumerate().take(words.len()) {
                let i = start + k;
                // Distance from the window's nearer edge: larger = more context on both sides.
                let edge = k.min(words.len() - 1 - k);
                let score = usize::MAX - edge;
                if score < best[*si][i] || best[*si][i] == usize::MAX {
                    best[*si][i] = score;
                    out[*si][i] = v;
                }
            }
        }
    }
    out
}

struct Head {
    layers: Vec<Linear>,
}

impl Head {
    fn new(vb: VarBuilder, input: usize, hidden: Option<usize>, tags: usize) -> candle_core::Result<Head> {
        let layers = match hidden {
            None => vec![linear(input, tags, vb.pp("out"))?],
            Some(h) => vec![linear(input, h, vb.pp("hidden"))?, linear(h, tags, vb.pp("out"))?],
        };
        Ok(Head { layers })
    }
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mut x = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            x = l.forward(&x)?;
            if i + 1 < self.layers.len() { x = x.relu()? }
        }
        Ok(x)
    }
}

/// Feature rows: the vector, optionally followed by a one-hot of a tagger's guess.
fn features(vecs: &[Vec<Vec<f32>>], guesses: Option<&[Vec<usize>]>, tags: usize) -> Vec<Vec<f32>> {
    let mut rows = Vec::new();
    for (si, s) in vecs.iter().enumerate() {
        for (i, v) in s.iter().enumerate() {
            let mut row = v.clone();
            if let Some(g) = guesses {
                let mut one = vec![0.0; tags];
                one[g[si][i]] = 1.0;
                row.extend(one);
            }
            rows.push(row);
        }
    }
    rows
}

fn train_and_score(name: &str, train: (&[Vec<f32>], &[u32]), evals: &[(&str, &[Vec<f32>], &[u32])], hidden: Option<usize>, tags: usize) {
    let dev = Device::Cpu;
    let input = train.0[0].len();
    let varmap = VarMap::new();
    let head = Head::new(VarBuilder::from_varmap(&varmap, DType::F32, &dev), input, hidden, tags).unwrap();
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: 2e-3, weight_decay: 1e-4, ..Default::default() }).unwrap();
    let n = train.0.len();
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = 0x9E3779B97F4A7C15u64;
    for _epoch in 0..8 {
        for i in (1..n).rev() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        for chunk in order.chunks(512) {
            let x: Vec<f32> = chunk.iter().flat_map(|&i| train.0[i].iter().copied()).collect();
            let y: Vec<u32> = chunk.iter().map(|&i| train.1[i]).collect();
            let x = Tensor::from_vec(x, (chunk.len(), input), &dev).unwrap();
            let y = Tensor::new(y.as_slice(), &dev).unwrap();
            let loss = candle_nn::loss::cross_entropy(&head.forward(&x).unwrap(), &y).unwrap();
            opt.backward_step(&loss).unwrap();
        }
    }
    let mut line = format!("{name:<11}");
    for (label, xs, ys) in evals {
        let mut right = 0usize;
        for (cx, cy) in xs.chunks(2048).zip(ys.chunks(2048)) {
            let x: Vec<f32> = cx.iter().flat_map(|r| r.iter().copied()).collect();
            let pred = head.forward(&Tensor::from_vec(x, (cx.len(), input), &dev).unwrap()).unwrap()
                .argmax(D::Minus1).unwrap().to_vec1::<u32>().unwrap();
            right += pred.iter().zip(cy.iter()).filter(|(p, y)| p == y).count();
        }
        line += &format!("  {label} {:.2}%", 100.0 * right as f64 / ys.len() as f64);
    }
    println!("{line}");
}

fn main() {
    let usage_name = std::env::args().nth(1).unwrap_or_else(|| "usage".into());
    let column = match std::env::args().nth(2).as_deref() { Some("xpos") => Column::Xpos, _ => Column::Upos };
    let load = |s: &str| conllu::parse(&std::fs::read_to_string(format!("data/en_ewt-ud-{s}.conllu")).unwrap());
    let (train, dev, test) = (load("train"), load("dev"), load("test"));
    let lex = Lexicon::load(Path::new("models/lexicon.bin")).unwrap();
    let model = UsageModel::load(&Path::new("models").join(&usage_name), Device::Cpu).unwrap();

    let mut tags: Vec<String> = train.iter().flatten().map(|t| column.of(t).to_string()).collect();
    tags.sort();
    tags.dedup();
    let index = |t: &str| tags.iter().position(|x| x == t).unwrap_or(0);
    let gold = |set: &[Sentence]| -> Vec<u32> { set.iter().flatten().map(|t| index(column.of(t)) as u32).collect() };

    // The shipped perceptron on dev/test; jackknifed (5 folds) on train.
    let tagger = Tagger::train(&train, column, Some(&lex), 5);
    let guess = |t: &Tagger, set: &[Sentence]| -> Vec<Vec<usize>> {
        set.iter().map(|s| {
            let words: Vec<&str> = s.iter().map(|t| t.form.as_str()).collect();
            t.tag(&words, Some(&lex)).into_iter().map(index).collect()
        }).collect()
    };
    let mut train_guess = vec![Vec::new(); train.len()];
    for fold in 0..5 {
        let rest: Vec<Sentence> = train.iter().enumerate().filter(|(i, _)| i % 5 != fold).map(|(_, s)| s.clone()).collect();
        let held: Vec<(usize, Sentence)> = train.iter().enumerate().filter(|(i, _)| i % 5 == fold).map(|(i, s)| (i, s.clone())).collect();
        let t = Tagger::train(&rest, column, Some(&lex), 5);
        let sents: Vec<Sentence> = held.iter().map(|(_, s)| s.clone()).collect();
        for ((i, _), g) in held.iter().zip(guess(&t, &sents)) { train_guess[*i] = g }
    }
    let (dev_guess, test_guess) = (guess(&tagger, &dev), guess(&tagger, &test));
    let acc = |g: &[Vec<usize>], set: &[Sentence]| -> f64 {
        let y = gold(set);
        let p: Vec<u32> = g.iter().flatten().map(|&x| x as u32).collect();
        100.0 * p.iter().zip(&y).filter(|(a, b)| a == b).count() as f64 / y.len() as f64
    };
    println!("{} tags; perceptron  dev {:.2}%  test {:.2}%  (train, jackknifed: {:.2}%)", tags.len(), acc(&dev_guess, &dev), acc(&test_guess, &test), acc(&train_guess, &train));

    eprintln!("usage model vectors ({usage_name})");
    let (vtr, vdev, vte) = (vectors(&model, &train), vectors(&model, &dev), vectors(&model, &test));
    let (ytr, ydev, yte) = (gold(&train), gold(&dev), gold(&test));
    let n = tags.len();
    for (name, hidden, with_tagger) in [("linear", None, false), ("mlp", Some(256), false), ("mlp+tagger", Some(256), true)] {
        let (ftr, fdev, fte) = if with_tagger {
            (features(&vtr, Some(&train_guess), n), features(&vdev, Some(&dev_guess), n), features(&vte, Some(&test_guess), n))
        } else {
            (features(&vtr, None, n), features(&vdev, None, n), features(&vte, None, n))
        };
        train_and_score(name, (&ftr, &ytr), &[("dev", &fdev, &ydev), ("test", &fte, &yte)], hidden, n);
    }
}
