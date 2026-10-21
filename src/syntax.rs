//! Neural syntax: UPOS tags and dependency arcs from the usage model
//! fine-tuned together with a tagging head and a biaffine parser (Dozat &
//! Manning), stacked on the shipped perceptron tagger and transition parser
//! (their choices are inputs). Trained by examples/parse_finetune.rs, which
//! uses these definitions; saved as a directory holding the fine-tuned usage
//! model plus parser_head.safetensors / parser_head.json.
//!
//! UD English EWT test (gold tokenization, predicted tags): UPOS and LAS in
//! parser_head.json; at the time of writing UAS 82.95% / LAS 79.52% vs the
//! transition parser's 81.66% / 77.54%.
use crate::parser::Arc;
use crate::usage::model::UsageModel;
use crate::usage::vocab::PAD;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder, VarMap};
use std::path::Path;

pub const WINDOW: usize = 40;
pub const STRIDE: usize = 20;
pub const CHAR_BUCKETS: usize = 1 << 16;
pub const CHAR_DIM: usize = 64;
pub const ARC: usize = 256;
pub const LAB: usize = 128;


pub fn spelling(word: &str) -> Vec<u32> {
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
pub struct Prepared {
    pub windows: Vec<Vec<u32>>,
    pub source: Vec<(usize, usize)>,
}

pub fn prepare(model: &UsageModel, forms: &[String]) -> Prepared {
    let words: Vec<String> = forms.iter().map(|w| w.to_lowercase()).collect();
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

pub struct Model {
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
pub type Prior = Vec<(usize, u32)>;

impl Model {
    pub fn new(vb: VarBuilder, input: usize, labels: usize, tags: usize, stack: bool) -> candle_core::Result<Model> {
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

pub fn tag_logits(m: &Model, xs: &[Tensor]) -> candle_core::Result<Tensor> {
    let x = Tensor::cat(xs, 0)?;
    m.tag_out.forward(&m.tag_hidden.forward(&x)?.relu()?)
}

/// Word matrices [n, input] for a batch of sentences, from one encoder pass over all their windows.
pub fn word_vectors(enc: &UsageModel, m: &Model, batch: &[(&[String], &Prepared, &Vec<usize>)], tags: usize, dev: &Device)
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
            for f in spelling(t) { ids.push(f); owner.push(k as u32) }
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
pub fn scores(m: &Model, xs: &[Tensor], priors: Option<&[&Prior]>, dev: &Device) -> candle_core::Result<(Tensor, Tensor, Tensor, usize)> {
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
pub fn label_logits(m: &Model, ld: &Tensor, lh: &Tensor, triples: &[(usize, usize, usize)], l: usize, prior_labels: Option<(&[u32], usize)>, dev: &Device) -> candle_core::Result<Tensor> {
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

/// A trained syntax model, ready to refine a sentence's tags and parse.
pub struct SyntaxModel {
    encoder: UsageModel,
    model: Model,
    _vars: VarMap,
    labels: Vec<String>,
    upos: Vec<String>,
    device: Device,
}

impl SyntaxModel {
    /// Loads `dir` (fine-tuned usage model + parser_head.safetensors/.json).
    pub fn load(dir: &Path) -> Result<SyntaxModel, Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("parser_head.json"))?)?;
        let strings = |k: &str| -> Vec<String> {
            meta[k].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default()
        };
        let (labels, upos) = (strings("labels"), strings("upos"));
        if labels.is_empty() || upos.is_empty() || meta["stack"] != serde_json::Value::Bool(true) {
            return Err("syntax model: missing labels/upos or not a stacked model".into());
        }
        let encoder = UsageModel::load(dir, device.clone())?;
        let mut vars = VarMap::new();
        let input = encoder.config.dim + CHAR_DIM + upos.len();
        let model = Model::new(VarBuilder::from_varmap(&vars, DType::F32, &device), input, labels.len(), upos.len(), true)?;
        vars.load(dir.join("parser_head.safetensors"))?;
        Ok(SyntaxModel { encoder, model, _vars: vars, labels, upos, device })
    }

    /// Refines the perceptron's UPOS tags and the transition parser's arcs
    /// for one sentence (both are inputs, as in training).
    pub fn refine(&self, words: &[&str], upos: &[&str], arcs: &[Arc]) -> candle_core::Result<(Vec<String>, Vec<Arc>)> {
        let n = words.len();
        if n == 0 { return Ok((vec![], vec![])) }
        let forms: Vec<String> = words.iter().map(|w| w.to_string()).collect();
        let prepared = prepare(&self.encoder, &forms);
        let tag_ix: Vec<usize> = upos.iter().map(|t| self.upos.iter().position(|u| u == t).unwrap_or(0)).collect();
        let label_ix = |l: &str| self.labels.iter().position(|x| x == l).unwrap_or(0) as u32;
        let prior: Prior = arcs.iter().map(|a| (a.head.min(n), label_ix(&a.label))).collect();
        let xs = word_vectors(&self.encoder, &self.model, &[(&forms[..], &prepared, &tag_ix)], self.upos.len(), &self.device)?;
        let tags = tag_logits(&self.model, &xs)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
        let (arc_scores, ld, lh, l) = scores(&self.model, &xs, Some(&[&prior][..]), &self.device)?;
        let heads = arc_scores.argmax(D::Minus1)?.to_vec2::<u32>()?.remove(0);
        let triples: Vec<(usize, usize, usize)> = (0..n).map(|k| (0, k, heads[k] as usize)).collect();
        let prior_labels: Vec<u32> = prior.iter().map(|&(_, l)| l).collect();
        let labs = label_logits(&self.model, &ld, &lh, &triples, l, Some((&prior_labels[..], self.labels.len())), &self.device)?
            .argmax(D::Minus1)?.to_vec1::<u32>()?;
        let tags = tags.into_iter().map(|t| self.upos[t as usize].clone()).collect();
        let arcs = (0..n).map(|k| Arc { head: heads[k] as usize, label: self.labels[labs[k] as usize].clone() }).collect();
        Ok((tags, arcs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spelling_features_cover_endings_beginnings_and_shape() {
        let f = spelling("Walked");
        assert!(f.len() >= 9);
        assert!(f.iter().all(|&b| (b as usize) < CHAR_BUCKETS));
        assert_ne!(spelling("walked"), spelling("Walked"), "shape (capitalization) is a feature");
    }

    /// Runs only where a trained model exists (`models/syntax`, or the
    /// training output directory).
    #[test]
    fn refine_returns_a_tag_and_an_arc_per_word() {
        let dir = ["models/syntax", "models/syntax-usage-morph-stacked"].into_iter().map(Path::new).find(|d| d.join("parser_head.json").exists());
        let Some(Ok(m)) = dir.map(SyntaxModel::load) else { return };
        let words = ["The", "dog", "barked", "."];
        let upos = ["DET", "NOUN", "VERB", "PUNCT"];
        let arcs = vec![Arc { head: 2, label: "det".into() }, Arc { head: 3, label: "nsubj".into() }, Arc { head: 0, label: "root".into() }, Arc { head: 3, label: "punct".into() }];
        let (tags, out) = m.refine(&words, &upos, &arcs).unwrap();
        assert_eq!((tags.len(), out.len()), (4, 4));
        assert!(out.iter().all(|a| a.head <= 4));
        assert_eq!(tags[1], "NOUN");
        assert_eq!(out[2].head, 0, "the verb is the root");
    }
}
