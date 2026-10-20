//! Fine-tunes the usage model together with a tagging head on UD English EWT
//! (train), scoring dev and test against the shipped perceptron tagger.
//!
//!   cargo run --release --features metal --example tag_finetune [usage-dir] [upos|xpos] [epochs] [--stack]
//!
//! --stack adds the perceptron's tag (jackknifed on train) as a one-hot input.
//! --chars adds spelling features (endings, beginnings, shape), hashed into a
//! learned embedding bag: what an unfamiliar word looks like, which the usage
//! model's vocabulary can't tell it.
//! Nothing is saved: this measures whether fine-tuning beats the shipped tagger.
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, linear, AdamW, Embedding, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use scribe::conllu::{self, Sentence};
use scribe::tagger::{Column, Tagger};
use scribe::usage::model::UsageModel;
use scribe::usage::vocab::PAD;
use scribe::Lexicon;
use std::path::Path;

const WINDOW: usize = 40;
const STRIDE: usize = 20;
const CHAR_BUCKETS: usize = 1 << 16;
const CHAR_DIM: usize = 64;

/// Hashed spelling features of a word: endings 1–4, beginnings 1–3, shape.
fn spelling(word: &str) -> Vec<u32> {
    let chars: Vec<char> = word.chars().collect();
    let lower: Vec<char> = word.to_lowercase().chars().collect();
    let mut feats: Vec<String> = Vec::new();
    for k in 1..=4.min(lower.len()) { feats.push(format!("s{k}:{}", lower[lower.len() - k..].iter().collect::<String>())) }
    for k in 1..=3.min(lower.len()) { feats.push(format!("p{k}:{}", lower[..k].iter().collect::<String>())) }
    let shape: String = chars.iter().map(|c| if c.is_uppercase() { 'X' } else if c.is_lowercase() { 'x' } else if c.is_ascii_digit() { 'd' } else { *c }).collect();
    let mut short = String::new();
    for c in shape.chars() { if !short.ends_with(c) { short.push(c) } }
    feats.push(format!("shape:{short}"));
    feats.push(format!("len:{}", chars.len().min(10)));
    feats.iter().map(|f| {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in f.bytes() { h = (h ^ b as u64).wrapping_mul(0x100000001b3) }
        (h % CHAR_BUCKETS as u64) as u32
    }).collect()
}

/// One training/eval unit: a window of a sentence.
struct Item {
    sentence: usize,
    start: usize,
    ids: Vec<u32>,
    /// (word index within the window, base-piece position)
    words: Vec<(usize, usize)>,
}

fn items(model: &UsageModel, sentences: &[Sentence]) -> Vec<Item> {
    let mut out = Vec::new();
    for (si, s) in sentences.iter().enumerate() {
        let words: Vec<String> = s.iter().map(|t| t.form.to_lowercase()).collect();
        let mut start = 0;
        loop {
            let end = (start + WINDOW).min(words.len());
            let (ids, base) = model.encode_words(&words[start..end]);
            out.push(Item { sentence: si, start, ids, words: base.into_iter().enumerate().collect() });
            if end == words.len() { break }
            start += STRIDE;
        }
    }
    out
}

struct Head(Vec<Linear>, Option<Embedding>);
impl Head {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.0[0].forward(x)?.relu()?;
        self.0[1].forward(&h)
    }
}

/// Logits for each word of each item in the batch, with the stacked one-hots if any.
fn batch_logits(model: &UsageModel, head: &Head, batch: &[&Item], stack: Option<&[Vec<usize>]>, tags: usize, dev: &Device,
                sentences: &[Sentence])
    -> candle_core::Result<(Tensor, Vec<(usize, usize)>)> {
    let len = batch.iter().map(|it| it.ids.len()).max().unwrap_or(1).max(1);
    let mut flat = Vec::with_capacity(batch.len() * len);
    for it in batch {
        flat.extend(it.ids.iter().copied());
        flat.extend(std::iter::repeat_n(PAD, len - it.ids.len()));
    }
    let hidden = model.hidden_states(&Tensor::from_vec(flat, (batch.len(), len), dev)?)?;
    let dim = hidden.dim(2)?;
    let hidden = hidden.reshape((batch.len() * len, dim))?;
    let mut rows = Vec::new();
    let mut words = Vec::new(); // (sentence, word index)
    for (b, it) in batch.iter().enumerate() {
        for &(w, piece) in &it.words {
            rows.push((b * len + piece) as u32);
            words.push((it.sentence, it.start + w));
        }
    }
    let mut x = hidden.index_select(&Tensor::new(rows.as_slice(), dev)?, 0)?;
    if let Some(g) = stack {
        let mut one = vec![0f32; words.len() * tags];
        for (k, (s, i)) in words.iter().enumerate() {
            one[k * tags + g[*s][*i]] = 1.0;
        }
        x = Tensor::cat(&[x, Tensor::from_vec(one, (words.len(), tags), dev)?], 1)?;
    }
    if let Some(emb) = &head.1 {
        // Mean of the word's spelling-feature embeddings.
        let mut ids = Vec::new();
        let mut owner = Vec::new();
        for (k, (s, i)) in words.iter().enumerate() {
            for f in spelling(&sentences[*s][*i].form) { ids.push(f); owner.push(k as u32) }
        }
        let e = emb.forward(&Tensor::new(ids.as_slice(), dev)?)?;
        let mut counts = vec![0f32; words.len()];
        for &o in &owner { counts[o as usize] += 1.0 }
        let sum = Tensor::zeros((words.len(), CHAR_DIM), DType::F32, dev)?.index_add(&Tensor::new(owner.as_slice(), dev)?, &e, 0)?;
        let mean = sum.broadcast_div(&Tensor::from_vec(counts, (words.len(), 1), dev)?)?;
        x = Tensor::cat(&[x, mean], 1)?;
    }
    Ok((head.forward(&x)?, words))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage_name = args.get(1).cloned().unwrap_or_else(|| "usage".into());
    let column = if args.get(2).map(String::as_str) == Some("xpos") { Column::Xpos } else { Column::Upos };
    let epochs: usize = args.get(3).and_then(|e| e.parse().ok()).unwrap_or(4);
    let stack = args.iter().any(|a| a == "--stack");
    let chars = args.iter().any(|a| a == "--chars");
    #[cfg(feature = "metal")]
    let dev = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let dev = Device::Cpu;

    let load = |s: &str| conllu::parse(&std::fs::read_to_string(format!("data/en_ewt-ud-{s}.conllu")).unwrap());
    let (train, devset, test) = (load("train"), load("dev"), load("test"));
    let lex = Lexicon::load(Path::new("models/lexicon.bin"))?;
    let model = UsageModel::load(&Path::new("models").join(&usage_name), dev.clone())?;
    let mut tags: Vec<String> = train.iter().flatten().map(|t| column.of(t).to_string()).collect();
    tags.sort();
    tags.dedup();
    let n = tags.len();
    let index = |t: &str| tags.iter().position(|x| x == t).unwrap_or(0);

    // Perceptron guesses: shipped-style on dev/test, jackknifed on train.
    let guess = |t: &Tagger, set: &[Sentence]| -> Vec<Vec<usize>> {
        set.iter().map(|s| {
            let w: Vec<&str> = s.iter().map(|t| t.form.as_str()).collect();
            t.tag(&w, Some(&lex)).into_iter().map(index).collect()
        }).collect()
    };
    let tagger = Tagger::train(&train, column, Some(&lex), 5);
    let (dev_guess, test_guess) = (guess(&tagger, &devset), guess(&tagger, &test));
    let mut train_guess = vec![Vec::new(); train.len()];
    if stack {
        for fold in 0..5 {
            let rest: Vec<Sentence> = train.iter().enumerate().filter(|(i, _)| i % 5 != fold).map(|(_, s)| s.clone()).collect();
            let held: Vec<usize> = (0..train.len()).filter(|i| i % 5 == fold).collect();
            let t = Tagger::train(&rest, column, Some(&lex), 5);
            let sents: Vec<Sentence> = held.iter().map(|&i| train[i].clone()).collect();
            for (&i, g) in held.iter().zip(guess(&t, &sents)) { train_guess[i] = g }
        }
    }
    let score = |g: &[Vec<usize>], set: &[Sentence]| -> f64 {
        let (mut r, mut t) = (0, 0);
        for (gs, s) in g.iter().zip(set) {
            for (gi, tok) in gs.iter().zip(s) { r += (*gi == index(column.of(tok))) as usize; t += 1 }
        }
        100.0 * r as f64 / t as f64
    };
    println!("{n} tags; perceptron dev {:.2}% test {:.2}%", score(&dev_guess, &devset), score(&test_guess, &test));

    let (tr_items, dev_items, te_items) = (items(&model, &train), items(&model, &devset), items(&model, &test));
    let dim = model.config.dim;
    let head_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&head_map, DType::F32, &dev);
    let input = dim + if stack { n } else { 0 } + if chars { CHAR_DIM } else { 0 };
    let head = Head(vec![linear(input, 256, vb.pp("hidden"))?, linear(256, n, vb.pp("out"))?],
                    if chars { Some(embedding(CHAR_BUCKETS, CHAR_DIM, vb.pp("chars"))?) } else { None });
    let mut enc_opt = AdamW::new(model.trainable_vars(), ParamsAdamW { lr: 1e-4, weight_decay: 0.01, ..Default::default() })?;
    let mut head_opt = AdamW::new(head_map.all_vars(), ParamsAdamW { lr: 1e-3, weight_decay: 1e-4, ..Default::default() })?;
    let gold_of = |set: &[Sentence], words: &[(usize, usize)]| -> Vec<u32> {
        words.iter().map(|&(s, i)| index(column.of(&set[s][i])) as u32).collect()
    };

    let evaluate = |items: &[Item], set: &[Sentence], g: &[Vec<usize>]| -> candle_core::Result<f64> {
        // Each word takes the prediction from the window where it's most central.
        let mut pred: Vec<Vec<(usize, u32)>> = set.iter().map(|s| vec![(usize::MAX, 0); s.len()]).collect();
        for chunk in items.chunks(64) {
            let batch: Vec<&Item> = chunk.iter().collect();
            let (logits, words) = batch_logits(&model, &head, &batch, stack.then_some(g), n, &dev, set)?;
            let best = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
            let mut k = 0;
            for it in &batch {
                let wlen = it.words.len();
                for (j, &(w, _)) in it.words.iter().enumerate() {
                    let edge = j.min(wlen - 1 - j);
                    let (s, i) = (it.sentence, it.start + w);
                    let slot = &mut pred[s][i];
                    if slot.0 == usize::MAX || edge > usize::MAX - slot.0 { *slot = (usize::MAX - edge, best[k]) }
                    k += 1;
                }
            }
            let _ = words;
        }
        let (mut r, mut t) = (0, 0);
        for (ps, s) in pred.iter().zip(set) {
            for (p, tok) in ps.iter().zip(s) { r += (p.1 as usize == index(column.of(tok))) as usize; t += 1 }
        }
        Ok(100.0 * r as f64 / t as f64)
    };

    let mut order: Vec<usize> = (0..tr_items.len()).collect();
    let mut rng = 0x2545F4914F6CDD1Du64;
    for epoch in 0..epochs {
        for i in (1..order.len()).rev() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            order.swap(i, (rng % (i as u64 + 1)) as usize);
        }
        let t0 = std::time::Instant::now();
        for chunk in order.chunks(32) {
            let batch: Vec<&Item> = chunk.iter().map(|&i| &tr_items[i]).collect();
            let (logits, words) = batch_logits(&model, &head, &batch, stack.then_some(&train_guess[..]), n, &dev, &train)?;
            let y = Tensor::new(gold_of(&train, &words).as_slice(), &dev)?;
            let loss = candle_nn::loss::cross_entropy(&logits, &y)?;
            let grads = loss.backward()?;
            enc_opt.step(&grads)?;
            head_opt.step(&grads)?;
        }
        println!("epoch {}: dev {:.2}% ({:.0?})", epoch + 1, evaluate(&dev_items, &devset, &dev_guess)?, t0.elapsed());
    }
    println!("fine-tuned{}{}: dev {:.2}%  test {:.2}%", if stack { "+stack" } else { "" }, if chars { "+chars" } else { "" },
             evaluate(&dev_items, &devset, &dev_guess)?, evaluate(&te_items, &test, &test_guess)?);
    Ok(())
}
