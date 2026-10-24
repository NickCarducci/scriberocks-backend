//! ScribeIE's model: an edit tagger (after LaserTagger, Malmi et al. 2019).
//! For each word of a proposal it decides keep or delete, and for each gap
//! (before every word, and at the end) whether to insert one of the phrases
//! editors most often add. It reads the frozen word vectors of a style's
//! judge (BERT-format) — so it adds only two small linear heads to the app —
//! and learns from editors' revisions (IteraTeR, Wikipedia and Wikinews
//! clarity edits; arXiv text left out, as not commercially reusable).
use crate::bert::Bert;
use candle_core::{DType, Device, Tensor, Var, D};
use std::collections::HashMap;
use std::path::Path;

/// One word's vector with its neighbors': [previous, this, next] (zeros past the ends).
fn features(h: &[Vec<f32>], i: usize, dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(3 * dim);
    for j in [i.checked_sub(1), Some(i), Some(i + 1)] {
        match j.and_then(|j| h.get(j)) {
            Some(v) => out.extend_from_slice(v),
            None => out.extend(std::iter::repeat_n(0.0, dim)),
        }
    }
    out
}

/// Tags that turn `before` into `after` (token lists): which tokens to keep,
/// and what to insert in each of the n + 1 gaps (gap i = before token i,
/// gap n = at the end). Longest common subsequence alignment.
pub fn align(before: &[String], after: &[String]) -> (Vec<bool>, Vec<Vec<String>>) {
    let (n, m) = (before.len(), after.len());
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if before[i] == after[j] { lcs[i + 1][j + 1] + 1 } else { lcs[i + 1][j].max(lcs[i][j + 1]) };
        }
    }
    let (mut keep, mut inserts) = (vec![false; n], vec![Vec::new(); n + 1]);
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        if i < n && j < m && before[i] == after[j] {
            keep[i] = true;
            i += 1;
            j += 1;
        } else if j < m && (i == n || lcs[i][j + 1] >= lcs[i + 1][j]) {
            inserts[i].push(after[j].clone());
            j += 1;
        } else {
            i += 1;
        }
    }
    (keep, inserts)
}

/// `before` with the tags applied.
pub fn apply(before: &[String], keep: &[bool], inserts: &[Vec<String>]) -> Vec<String> {
    let mut out = Vec::new();
    for (i, w) in before.iter().enumerate() {
        out.extend(inserts[i].iter().cloned());
        if keep[i] { out.push(w.clone()) }
    }
    out.extend(inserts[before.len()].iter().cloned());
    out
}

fn ngram_counts(tokens: &[String], n: usize) -> HashMap<Vec<String>, f64> {
    let mut c = HashMap::new();
    if tokens.len() >= n {
        for w in tokens.windows(n) { *c.entry(w.to_vec()).or_insert(0.0) += 1.0 }
    }
    c
}

/// SARI (Xu et al. 2016) with one reference, n-grams 1–4: the mean of the
/// F1 of words kept, the F1 of words added and the precision of words
/// deleted, each judged against what the editor kept, added and deleted.
/// As in the reference implementation, adding or deleting nothing earns
/// nothing for that part. 0–100; copying the source unchanged scores
/// about a third of its keep score.
pub fn sari(source: &[String], prediction: &[String], reference: &[String]) -> f64 {
    let f1 = |p: f64, r: f64| if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
    let (mut keep, mut add, mut del) = (0.0, 0.0, 0.0);
    for n in 1..=4 {
        let (s, p, r) = (ngram_counts(source, n), ngram_counts(prediction, n), ngram_counts(reference, n));
        // Added: in the prediction (or reference) but not the source.
        let pa: Vec<_> = p.keys().filter(|g| !s.contains_key(*g)).collect();
        let ra: Vec<_> = r.keys().filter(|g| !s.contains_key(*g)).collect();
        let hit = pa.iter().filter(|g| r.contains_key(**g)).count() as f64;
        let (ap, ar) = (if pa.is_empty() { 0.0 } else { hit / pa.len() as f64 }, if ra.is_empty() { 0.0 } else { hit / ra.len() as f64 });
        add += f1(ap, ar);
        // Kept: source n-grams the prediction keeps, against what the reference keeps.
        let (mut k_hit, mut k_pred, mut k_ref) = (0.0, 0.0, 0.0);
        let (mut d_hit, mut d_pred) = (0.0, 0.0);
        for (g, &sc) in &s {
            let (pc, rc) = (p.get(g).copied().unwrap_or(0.0).min(sc), r.get(g).copied().unwrap_or(0.0).min(sc));
            k_pred += pc;
            k_ref += rc;
            k_hit += pc.min(rc);
            let (pd, rd) = (sc - pc, sc - rc);
            d_pred += pd;
            d_hit += pd.min(rd);
        }
        let (kp, kr) = (if k_pred > 0.0 { k_hit / k_pred } else { 0.0 }, if k_ref > 0.0 { k_hit / k_ref } else { 0.0 });
        keep += f1(kp, kr);
        del += if d_pred > 0.0 { d_hit / d_pred } else { 0.0 };
    }
    100.0 * (keep + add + del) / 12.0
}

/// One training sentence: word vectors, gold keep flags, gold insert ids
/// (0 = nothing, else 1 + index into the phrase list) for each of the n + 1 gaps.
pub struct Example {
    pub vectors: Vec<Vec<f32>>,
    pub keep: Vec<bool>,
    pub inserts: Vec<usize>,
}

pub struct EditTagger {
    pub phrases: Vec<String>,
    dim: usize,
    /// [3·dim, 2] and [2]: delete head (class 1 = delete).
    del_w: Tensor,
    del_b: Tensor,
    /// [3·dim, phrases + 1] and [phrases + 1]: insert head (class 0 = nothing).
    ins_w: Tensor,
    ins_b: Tensor,
}

impl EditTagger {
    /// Softmax regression for both heads, Adam, on the CPU.
    pub fn train(examples: &[Example], phrases: Vec<String>, dim: usize, epochs: usize, lr: f64, seed: u64) -> candle_core::Result<EditTagger> {
        let dev = Device::Cpu;
        let f = 3 * dim;
        let k = phrases.len() + 1;
        let vars = [
            ("del_w", Var::zeros((f, 2), DType::F32, &dev)?),
            ("del_b", Var::zeros(2, DType::F32, &dev)?),
            ("ins_w", Var::zeros((f, k), DType::F32, &dev)?),
            ("ins_b", Var::zeros(k, DType::F32, &dev)?),
        ];
        let mut opt = crate::slm_train::Adam::new(vars.iter().map(|(n, v)| (n.to_string(), v.clone())).collect(), candle_nn::ParamsAdamW { lr, weight_decay: 0.0, ..Default::default() })?;
        let mut rng = crate::slm_train::Rng(seed | 1);
        let mut order: Vec<usize> = (0..examples.len()).collect();
        for epoch in 0..epochs {
            for i in (1..order.len()).rev() { order.swap(i, (rng.next() % (i as u64 + 1)) as usize) }
            let mut total = 0.0;
            for chunk in order.chunks(64) {
                let (mut dx, mut dy, mut ix, mut iy) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for &e in chunk {
                    let ex = &examples[e];
                    for i in 0..ex.vectors.len() {
                        dx.extend(features(&ex.vectors, i, dim));
                        dy.push(u32::from(!ex.keep[i]));
                    }
                    // Gap i sits before word i: read [word i-1, word i, word i+1] as for word i;
                    // the end gap reads past the last word.
                    for (g, &ins) in ex.inserts.iter().enumerate() {
                        ix.extend(features(&ex.vectors, g, dim));
                        iy.push(ins as u32);
                    }
                }
                let (nd, ni) = (dy.len(), iy.len());
                let dl = candle_nn::loss::cross_entropy(&Tensor::from_vec(dx, (nd, f), &dev)?.matmul(vars[0].1.as_tensor())?.broadcast_add(vars[1].1.as_tensor())?, &Tensor::from_vec(dy, nd, &dev)?)?;
                let il = candle_nn::loss::cross_entropy(&Tensor::from_vec(ix, (ni, f), &dev)?.matmul(vars[2].1.as_tensor())?.broadcast_add(vars[3].1.as_tensor())?, &Tensor::from_vec(iy, ni, &dev)?)?;
                let loss = (dl + il)?;
                total += loss.to_scalar::<f32>()? as f64 * chunk.len() as f64;
                opt.step(&loss.backward()?)?;
            }
            eprintln!("  epoch {}: loss {:.4}", epoch + 1, total / examples.len().max(1) as f64);
        }
        Ok(EditTagger { phrases, dim, del_w: vars[0].1.as_tensor().clone(), del_b: vars[1].1.as_tensor().clone(), ins_w: vars[2].1.as_tensor().clone(), ins_b: vars[3].1.as_tensor().clone() })
    }

    /// Edits for one sentence's word vectors: a word is deleted when the
    /// delete head is at least `confidence` sure, and a phrase inserted when
    /// the insert head gives it at least `confidence`.
    pub fn tag(&self, vectors: &[Vec<f32>], confidence: f32) -> candle_core::Result<(Vec<bool>, Vec<Vec<String>>)> {
        let n = vectors.len();
        let rows: Vec<f32> = (0..=n).flat_map(|i| features(vectors, i, self.dim)).collect();
        let x = Tensor::from_vec(rows, (n + 1, 3 * self.dim), &Device::Cpu)?;
        let pd = candle_nn::ops::softmax(&x.narrow(0, 0, n)?.matmul(&self.del_w)?.broadcast_add(&self.del_b)?, D::Minus1)?.to_vec2::<f32>()?;
        let pi = candle_nn::ops::softmax(&x.matmul(&self.ins_w)?.broadcast_add(&self.ins_b)?, D::Minus1)?.to_vec2::<f32>()?;
        let keep = pd.iter().map(|p| p[1] < confidence).collect();
        let inserts = pi.iter().map(|p| {
            let (best, &pb) = p.iter().enumerate().skip(1).max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
            if pb >= confidence { self.phrases[best - 1].split(' ').map(String::from).collect() } else { vec![] }
        }).collect();
        Ok((keep, inserts))
    }

    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;
        let w: HashMap<String, Tensor> = [("del_w", &self.del_w), ("del_b", &self.del_b), ("ins_w", &self.ins_w), ("ins_b", &self.ins_b)].into_iter().map(|(n, t)| (n.to_string(), t.clone())).collect();
        candle_core::safetensors::save(&w, dir.join("tagger.safetensors"))?;
        std::fs::write(dir.join("tagger.json"), serde_json::to_string(&serde_json::json!({ "dim": self.dim, "phrases": self.phrases }))?)?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<EditTagger, Box<dyn std::error::Error>> {
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tagger.json"))?)?;
        let mut w = candle_core::safetensors::load(dir.join("tagger.safetensors"), &Device::Cpu)?;
        let mut take = |n: &str| w.remove(n).ok_or_else(|| format!("{n} missing"));
        Ok(EditTagger {
            phrases: meta["phrases"].as_array().ok_or("no phrases")?.iter().filter_map(|p| p.as_str().map(String::from)).collect(),
            dim: meta["dim"].as_u64().ok_or("no dim")? as usize,
            del_w: take("del_w")?,
            del_b: take("del_b")?,
            ins_w: take("ins_w")?,
            ins_b: take("ins_b")?,
        })
    }
}

/// ScribeIE as the engine's Enhancer: a style's judge supplies the word vectors.
pub struct TaggerEnhancer {
    pub encoder: Bert,
    pub tagger: EditTagger,
    pub confidence: f32,
}

impl crate::scribe_ie::Enhancer for TaggerEnhancer {
    fn enhance(&self, _rock: &str, proposal: &str) -> Option<String> {
        let words = crate::text::tokenize(proposal);
        if words.is_empty() { return None }
        let vectors = self.encoder.contextual_batch(std::slice::from_ref(&words)).ok()?.into_iter().next()?;
        if vectors.len() != words.len() { return None }
        let (keep, inserts) = self.tagger.tag(&vectors, self.confidence).ok()?;
        let out = apply(&words, &keep, &inserts);
        (out != words).then(|| crate::text::detokenize(&out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    #[test]
    fn alignment_rebuilds_the_revision() {
        for (b, a) in [("the cat sat", "the black cat sat down"), ("we did use it", "we used it"), ("a b c", "x y z"), ("same", "same")] {
            let (keep, ins) = align(&t(b), &t(a));
            assert_eq!(apply(&t(b), &keep, &ins), t(a), "{b} → {a}");
        }
        let (keep, ins) = align(&t("we did use it"), &t("we used it"));
        assert_eq!(keep, vec![true, false, false, true]);
        assert_eq!(ins[1], t("used"));
    }

    #[test]
    fn sari_rewards_the_editors_edit_over_copying() {
        let (src, refr) = (t("the results is very good"), t("the results are good"));
        let copy = sari(&src, &src, &refr);
        let exact = sari(&src, &refr, &refr);
        // Under 100: the editor keeps no 4-gram here, and empty parts earn nothing.
        assert!(exact > 80.0, "matching the editor exactly: {exact}");
        assert!(copy < 34.0, "copying earns keep credit only: {copy}");
        assert_eq!(sari(&src, &src, &src), 100.0 / 3.0, "nothing to add or delete, nothing done: keep only");
        assert!(sari(&src, &t("the results are very good"), &refr) > copy);
    }

    #[test]
    fn a_trained_tagger_learns_a_simple_edit_and_survives_saving() {
        // Toy vectors: word "very" (vector [1,0]) is always deleted; nothing else changes.
        let ex = |very_at: usize| {
            let vectors: Vec<Vec<f32>> = (0..4).map(|i| if i == very_at { vec![1.0, 0.0] } else { vec![0.0, 1.0] }).collect();
            Example { keep: (0..4).map(|i| i != very_at).collect(), inserts: vec![0; 5], vectors }
        };
        let data: Vec<Example> = (0..40).map(|i| ex(i % 4)).collect();
        let tagger = EditTagger::train(&data, vec!["really".into()], 2, 60, 0.05, 7).unwrap();
        let (keep, inserts) = tagger.tag(&data[1].vectors, 0.5).unwrap();
        assert_eq!(keep, data[1].keep);
        assert!(inserts.iter().all(Vec::is_empty));
        let dir = std::env::temp_dir().join(format!("edit-tagger-{}", std::process::id()));
        tagger.save(&dir).unwrap();
        let back = EditTagger::load(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(back.tag(&data[1].vectors, 0.5).unwrap().0, data[1].keep);
    }
}
