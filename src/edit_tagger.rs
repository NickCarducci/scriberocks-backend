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

/// Whether a revision only rewords: it changes no name (a capitalized word
/// past the first) and no number. Fact edits (Wikinews trimming a name or a
/// date) teach deleting details, not clearer phrasing.
pub fn rewords(before: &[String], after: &[String]) -> bool {
    let (keep, inserts) = align(before, after);
    let facty = |i: usize, w: &str| w.chars().any(|c| c.is_ascii_digit()) || (i > 0 && w.chars().next().is_some_and(char::is_uppercase));
    let deleted = before.iter().enumerate().filter(|(i, _)| !keep[*i]).any(|(i, w)| facty(i, w));
    let added = inserts.iter().enumerate().any(|(g, ws)| ws.iter().enumerate().any(|(k, w)| facty(g + k, w)));
    !deleted && !added
}

/// Whether a revision writes something: it inserts or replaces at least one
/// word. Pure deletions are left out: in IteraTeR's Wikipedia and Wikinews
/// edits many are extraction artifacts (a linked word missing from the
/// "after" text: "It is endemic to the island" → "It is to the island").
pub fn writes(before: &[String], after: &[String]) -> bool {
    align(before, after).1.iter().any(|ins| ins.iter().any(|w| w.chars().any(char::is_alphanumeric)))
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
    /// Optional shared hidden layer ([3·dim, H] and [H], GELU) before both heads.
    hidden: Option<(Tensor, Tensor)>,
    /// [3·dim, 2] and [2]: delete head (class 1 = delete).
    del_w: Tensor,
    del_b: Tensor,
    /// [3·dim, phrases + 1] and [phrases + 1]: insert head (class 0 = nothing).
    ins_w: Tensor,
    ins_b: Tensor,
}

impl EditTagger {
    /// Both heads (softmax), with a shared GELU hidden layer of `hidden`
    /// units when nonzero; Adam, on the CPU.
    pub fn train(examples: &[Example], phrases: Vec<String>, dim: usize, hidden: usize, epochs: usize, lr: f64, seed: u64) -> candle_core::Result<EditTagger> {
        let dev = Device::Cpu;
        let f = 3 * dim;
        let k = phrases.len() + 1;
        let into = if hidden > 0 { hidden } else { f };
        let vars = [
            ("del_w", Var::zeros((into, 2), DType::F32, &dev)?),
            ("del_b", Var::zeros(2, DType::F32, &dev)?),
            ("ins_w", Var::zeros((into, k), DType::F32, &dev)?),
            ("ins_b", Var::zeros(k, DType::F32, &dev)?),
        ];
        let layer = match hidden {
            0 => None,
            h => Some((Var::from_tensor(&(Tensor::randn(0f32, 1.0, (f, h), &dev)? / (f as f64).sqrt())?)?, Var::zeros(h, DType::F32, &dev)?)),
        };
        let mut named: Vec<(String, Var)> = vars.iter().map(|(n, v)| (n.to_string(), v.clone())).collect();
        if let Some((w, b)) = &layer { named.extend([("h_w".to_string(), w.clone()), ("h_b".to_string(), b.clone())]) }
        let mut opt = crate::slm_train::Adam::new(named, candle_nn::ParamsAdamW { lr, weight_decay: 0.0, ..Default::default() })?;
        let lift = |x: Tensor| -> candle_core::Result<Tensor> {
            match &layer { Some((w, b)) => x.matmul(w.as_tensor())?.broadcast_add(b.as_tensor())?.gelu_erf(), None => Ok(x) }
        };
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
                let dl = candle_nn::loss::cross_entropy(&lift(Tensor::from_vec(dx, (nd, f), &dev)?)?.matmul(vars[0].1.as_tensor())?.broadcast_add(vars[1].1.as_tensor())?, &Tensor::from_vec(dy, nd, &dev)?)?;
                let il = candle_nn::loss::cross_entropy(&lift(Tensor::from_vec(ix, (ni, f), &dev)?)?.matmul(vars[2].1.as_tensor())?.broadcast_add(vars[3].1.as_tensor())?, &Tensor::from_vec(iy, ni, &dev)?)?;
                let loss = (dl + il)?;
                total += loss.to_scalar::<f32>()? as f64 * chunk.len() as f64;
                opt.step(&loss.backward()?)?;
            }
            eprintln!("  epoch {}: loss {:.4}", epoch + 1, total / examples.len().max(1) as f64);
        }
        let hidden = layer.map(|(w, b)| (w.as_tensor().clone(), b.as_tensor().clone()));
        Ok(EditTagger { phrases, dim, hidden, del_w: vars[0].1.as_tensor().clone(), del_b: vars[1].1.as_tensor().clone(), ins_w: vars[2].1.as_tensor().clone(), ins_b: vars[3].1.as_tensor().clone() })
    }

    /// As `train`, but the encoder learns too: word vectors come from
    /// `bert` (trainable, its weights in `bert_vars`) inside the gradient.
    /// The encoder moves at `encoder_lr`, the heads at `head_lr`. Sentences
    /// are (words, keep flags, insert ids for the n + 1 gaps); ones the
    /// encoder would truncate are skipped. Save the encoder with
    /// `bert_vars.save(...)`; the returned tagger holds the heads.
    #[allow(clippy::too_many_arguments)]
    pub fn train_finetuned(bert: &Bert, bert_vars: &candle_nn::VarMap, sentences: &[(Vec<String>, Vec<bool>, Vec<usize>)], phrases: Vec<String>,
                           hidden: usize, epochs: usize, encoder_lr: f64, head_lr: f64, seed: u64, dev: &Device) -> candle_core::Result<EditTagger> {
        let dim = bert.hidden(&[vec![0u32]])?.dim(2)?;
        let f = 3 * dim;
        let k = phrases.len() + 1;
        let into = if hidden > 0 { hidden } else { f };
        let heads = [
            ("del_w", Var::zeros((into, 2), DType::F32, dev)?),
            ("del_b", Var::zeros(2, DType::F32, dev)?),
            ("ins_w", Var::zeros((into, k), DType::F32, dev)?),
            ("ins_b", Var::zeros(k, DType::F32, dev)?),
        ];
        let layer = match hidden {
            0 => None,
            h => Some((Var::from_tensor(&(Tensor::randn(0f32, 1.0, (f, h), dev)? / (f as f64).sqrt())?)?, Var::zeros(h, DType::F32, dev)?)),
        };
        let mut named: Vec<(String, Var)> = heads.iter().map(|(n, v)| (n.to_string(), v.clone())).collect();
        if let Some((w, b)) = &layer { named.extend([("h_w".to_string(), w.clone()), ("h_b".to_string(), b.clone())]) }
        let mut head_opt = crate::slm_train::Adam::new(named, candle_nn::ParamsAdamW { lr: head_lr, weight_decay: 0.0, ..Default::default() })?;
        let enc: Vec<(String, Var)> = bert_vars.data().lock().unwrap().iter().map(|(n, v)| (n.clone(), v.clone())).collect();
        let mut enc_opt = crate::slm_train::Adam::new(enc, candle_nn::ParamsAdamW { lr: encoder_lr, weight_decay: 0.01, ..Default::default() })?;
        let encoded: Vec<Option<(Vec<u32>, Vec<usize>)>> = sentences.iter().map(|(w, _, _)| {
            let (ids, first) = bert.encode_words(w);
            (first.len() == w.len() && first.iter().all(|&x| x + 1 < ids.len())).then_some((ids, first))
        }).collect();
        let usable: Vec<usize> = (0..sentences.len()).filter(|&i| encoded[i].is_some()).collect();
        let mut rng = crate::slm_train::Rng(seed | 1);
        let mut order = usable.clone();
        for epoch in 0..epochs {
            for i in (1..order.len()).rev() { order.swap(i, (rng.next() % (i as u64 + 1)) as usize) }
            let (mut total, mut batches) = (0.0, 0);
            for chunk in order.chunks(32) {
                let rows: Vec<Vec<u32>> = chunk.iter().map(|&i| encoded[i].as_ref().unwrap().0.clone()).collect();
                let h = bert.hidden(&rows)?;
                let (b, t, _) = h.dims3()?;
                let table = Tensor::cat(&[h.reshape((b * t, dim))?, Tensor::zeros((1, dim), DType::F32, dev)?], 0)?;
                let zero = (b * t) as u32;
                let (mut dsel, mut isel) = ([Vec::new(), Vec::new(), Vec::new()], [Vec::new(), Vec::new(), Vec::new()]);
                let (mut dy, mut iy) = (Vec::new(), Vec::new());
                for (row, &i) in chunk.iter().enumerate() {
                    let first = &encoded[i].as_ref().unwrap().1;
                    let at = |w: Option<usize>| w.and_then(|w| first.get(w)).map_or(zero, |&x| (row * t + x) as u32);
                    let (_, keep, ins) = &sentences[i];
                    for w in 0..first.len() {
                        for (j, x) in [w.checked_sub(1), Some(w), Some(w + 1)].into_iter().enumerate() { dsel[j].push(at(x)) }
                        dy.push(u32::from(!keep[w]));
                    }
                    for (g, &id) in ins.iter().enumerate() {
                        for (j, x) in [g.checked_sub(1), Some(g), Some(g + 1)].into_iter().enumerate() { isel[j].push(at(x)) }
                        iy.push(id as u32);
                    }
                }
                let gather = |sel: &[Vec<u32>; 3]| -> candle_core::Result<Tensor> {
                    let parts = sel.iter().map(|s| table.index_select(&Tensor::new(s.as_slice(), dev)?, 0)).collect::<candle_core::Result<Vec<_>>>()?;
                    let x = Tensor::cat(&parts, 1)?;
                    match &layer { Some((w, b)) => x.matmul(w.as_tensor())?.broadcast_add(b.as_tensor())?.gelu_erf(), None => Ok(x) }
                };
                let (nd, ni) = (dy.len(), iy.len());
                let dl = candle_nn::loss::cross_entropy(&gather(&dsel)?.matmul(heads[0].1.as_tensor())?.broadcast_add(heads[1].1.as_tensor())?, &Tensor::from_vec(dy, nd, dev)?)?;
                let il = candle_nn::loss::cross_entropy(&gather(&isel)?.matmul(heads[2].1.as_tensor())?.broadcast_add(heads[3].1.as_tensor())?, &Tensor::from_vec(iy, ni, dev)?)?;
                let loss = (dl + il)?;
                let grads = loss.backward()?;
                head_opt.step(&grads)?;
                enc_opt.step(&grads)?;
                total += loss.to_scalar::<f32>()? as f64;
                batches += 1;
                if batches % 500 == 0 { eprintln!("  epoch {} batch {batches}: loss {:.4}", epoch + 1, total / batches as f64) }
            }
            eprintln!("  epoch {}: loss {:.4} ({} sentences)", epoch + 1, total / batches.max(1) as f64, order.len());
        }
        let cpu = |v: &Var| v.as_tensor().to_device(&Device::Cpu);
        Ok(EditTagger {
            phrases,
            dim,
            hidden: match &layer { Some((w, b)) => Some((cpu(w)?, cpu(b)?)), None => None },
            del_w: cpu(&heads[0].1)?, del_b: cpu(&heads[1].1)?, ins_w: cpu(&heads[2].1)?, ins_b: cpu(&heads[3].1)?,
        })
    }

    /// Edits for one sentence's word vectors: a word is deleted when the
    /// delete head is at least `confidence` sure, and a phrase inserted when
    /// the insert head gives it at least `confidence`.
    pub fn tag(&self, vectors: &[Vec<f32>], confidence: f32) -> candle_core::Result<(Vec<bool>, Vec<Vec<String>>)> {
        let (pd, pi) = self.probabilities(vectors)?;
        let keep = pd.iter().map(|p| p[1] < confidence).collect();
        let inserts = pi.iter().map(|p| {
            let (best, &pb) = p.iter().enumerate().skip(1).max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
            if pb >= confidence { self.phrases[best - 1].split(' ').map(String::from).collect() } else { vec![] }
        }).collect();
        Ok((keep, inserts))
    }

    /// Per word [keep, delete] and per gap [nothing, phrase 1, ...] probabilities.
    pub fn probabilities(&self, vectors: &[Vec<f32>]) -> candle_core::Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let n = vectors.len();
        let rows: Vec<f32> = (0..=n).flat_map(|i| features(vectors, i, self.dim)).collect();
        let x = Tensor::from_vec(rows, (n + 1, 3 * self.dim), &Device::Cpu)?;
        let x = match &self.hidden { Some((w, b)) => x.matmul(w)?.broadcast_add(b)?.gelu_erf()?, None => x };
        let pd = candle_nn::ops::softmax(&x.narrow(0, 0, n)?.matmul(&self.del_w)?.broadcast_add(&self.del_b)?, D::Minus1)?.to_vec2::<f32>()?;
        let pi = candle_nn::ops::softmax(&x.matmul(&self.ins_w)?.broadcast_add(&self.ins_b)?, D::Minus1)?.to_vec2::<f32>()?;
        Ok((pd, pi))
    }

    /// How editor-like turning `before` into `after` is: the mean log
    /// probability of each keep/delete and insert decision the edit makes.
    /// An insertion outside the phrase list counts as "some insertion" there.
    pub fn edit_log_likelihood(&self, vectors: &[Vec<f32>], before: &[String], after: &[String]) -> candle_core::Result<f32> {
        let (pd, pi) = self.probabilities(vectors)?;
        let (keep, inserts) = align(before, after);
        let index: HashMap<&str, usize> = self.phrases.iter().enumerate().map(|(i, p)| (p.as_str(), i + 1)).collect();
        let ln = |p: f32| p.max(1e-6).ln();
        let mut total = 0.0;
        for (w, &k) in keep.iter().enumerate() { total += ln(pd[w][usize::from(!k)]) }
        for (g, ins) in inserts.iter().enumerate() {
            total += match (ins.is_empty(), index.get(ins.join(" ").as_str())) {
                (true, _) => ln(pi[g][0]),
                (false, Some(&i)) => ln(pi[g][i]),
                (false, None) => ln(1.0 - pi[g][0]),
            };
        }
        Ok(total / (keep.len() + inserts.len()) as f32)
    }

    pub fn save(&self, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(dir)?;
        let mut w: HashMap<String, Tensor> = [("del_w", &self.del_w), ("del_b", &self.del_b), ("ins_w", &self.ins_w), ("ins_b", &self.ins_b)].into_iter().map(|(n, t)| (n.to_string(), t.clone())).collect();
        if let Some((hw, hb)) = &self.hidden { w.insert("h_w".into(), hw.clone()); w.insert("h_b".into(), hb.clone()); }
        candle_core::safetensors::save(&w, dir.join("tagger.safetensors"))?;
        std::fs::write(dir.join("tagger.json"), serde_json::to_string(&serde_json::json!({ "dim": self.dim, "phrases": self.phrases }))?)?;
        Ok(())
    }

    pub fn load(dir: &Path) -> Result<EditTagger, Box<dyn std::error::Error>> {
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tagger.json"))?)?;
        let mut w = candle_core::safetensors::load(dir.join("tagger.safetensors"), &Device::Cpu)?;
        let hidden = match (w.remove("h_w"), w.remove("h_b")) { (Some(a), Some(b)) => Some((a, b)), _ => None };
        let mut take = |n: &str| w.remove(n).ok_or_else(|| format!("{n} missing"));
        Ok(EditTagger {
            hidden,
            phrases: meta["phrases"].as_array().ok_or("no phrases")?.iter().filter_map(|p| p.as_str().map(String::from)).collect(),
            dim: meta["dim"].as_u64().ok_or("no dim")? as usize,
            del_w: take("del_w")?,
            del_b: take("del_b")?,
            ins_w: take("ins_w")?,
            ins_b: take("ins_b")?,
        })
    }
}

/// Words ScribeIE may never delete: negations (including "didnt"-style
/// spellings), numbers, and names (capitalized past the first word).
pub fn protected(i: usize, word: &str) -> bool {
    let w = word.to_lowercase();
    const NEGATIONS: [&str; 14] = ["not", "n't", "no", "never", "nor", "none", "nothing", "nobody", "neither", "nowhere", "cannot", "without", "hardly", "barely"];
    NEGATIONS.contains(&w.as_str())
        || (w.len() > 3 && w.ends_with("nt") && ["dont", "didnt", "doesnt", "cant", "wont", "isnt", "wasnt", "arent", "werent", "havent", "hasnt", "hadnt", "couldnt", "shouldnt", "wouldnt", "aint"].contains(&w.as_str()))
        || word.chars().any(|c| c.is_ascii_digit())
        || (i > 0 && word.chars().next().is_some_and(char::is_uppercase))
}

/// ScribeIE as the engine's Enhancer: its own encoder supplies the word vectors.
pub struct TaggerEnhancer {
    pub encoder: Bert,
    pub tagger: EditTagger,
    pub confidence: f32,
}

impl TaggerEnhancer {
    /// From a folder with `encoder/` (BERT-format, full or 8-bit), the
    /// tagger's heads, and `confidence.txt` (default 0.5).
    pub fn load(dir: &Path) -> Result<TaggerEnhancer, Box<dyn std::error::Error>> {
        Ok(TaggerEnhancer {
            encoder: Bert::load(&dir.join("encoder"), Device::Cpu)?,
            tagger: EditTagger::load(dir)?,
            confidence: std::fs::read_to_string(dir.join("confidence.txt")).ok().and_then(|c| c.trim().parse().ok()).unwrap_or(0.5),
        })
    }

    /// ScribeIE's rank for a proposal: how editor-like its edit of `original` is.
    pub fn rank(&self, original: &str, proposal: &str) -> Option<f32> {
        let (before, after) = (crate::text::tokenize(original), crate::text::tokenize(proposal));
        let vectors = self.encoder.contextual_batch(std::slice::from_ref(&before)).ok()?.into_iter().next()?;
        if vectors.len() != before.len() { return None }
        self.tagger.edit_log_likelihood(&vectors, &before, &after).ok()
    }
}

impl crate::scribe_ie::Enhancer for TaggerEnhancer {
    fn enhance(&self, _rock: &str, proposal: &str) -> Option<String> {
        let words = crate::text::tokenize(proposal);
        if words.is_empty() { return None }
        let vectors = self.encoder.contextual_batch(std::slice::from_ref(&words)).ok()?.into_iter().next()?;
        if vectors.len() != words.len() { return None }
        let (mut keep, inserts) = self.tagger.tag(&vectors, self.confidence).ok()?;
        // Never delete meaning: negations, numbers, names (as in its training data).
        for (i, w) in words.iter().enumerate() {
            if protected(i, w) { keep[i] = true }
        }
        let out = apply(&words, &keep, &inserts);
        (out != words).then(|| crate::text::detokenize(&out))
    }

    fn rank(&self, rock: &str, proposal: &str) -> Option<f32> {
        TaggerEnhancer::rank(self, rock, proposal)
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
    fn fine_tuning_learns_an_insertion_and_moves_the_encoder() {
        // Toy task: insert "really" before "good"; nothing else changes.
        let lines = ["the food is good .", "this film was good .", "her plan is good .", "the weather is bad .", "his car was bad ."];
        let mut vocab: Vec<String> = crate::wordpiece::SPECIALS.iter().map(|s| s.to_string()).collect();
        for w in lines.iter().flat_map(|l| l.split(' ')).chain(["really"]) { if !vocab.iter().any(|v| v == w) { vocab.push(w.to_string()) } }
        let dir = std::env::temp_dir().join(format!("edit-tagger-ft-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("vocab.txt"), vocab.join("\n")).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::json!({"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
            "intermediate_size": 32, "max_position_embeddings": 32, "layer_norm_eps": 1e-12, "pre_layer_norm": true}).to_string()).unwrap();
        let (bert, vars) = Bert::new_random(&dir, Device::Cpu).unwrap();
        let before = vars.all_vars()[0].as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let data: Vec<(Vec<String>, Vec<bool>, Vec<usize>)> = (0..8).flat_map(|_| lines.iter()).map(|l| {
            let w = t(l);
            let ins = (0..=w.len()).map(|g| usize::from(w.get(g).is_some_and(|x| x == "good"))).collect();
            (w.clone(), vec![true; w.len()], ins)
        }).collect();
        let tagger = EditTagger::train_finetuned(&bert, &vars, &data, vec!["really".into()], 16, 30, 1e-3, 1e-2, 3, &Device::Cpu).unwrap();
        let after = vars.all_vars()[0].as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(before.iter().zip(&after).any(|(a, b)| a != b), "the encoder trained");
        let words = t("the food is good .");
        let v = bert.contextual_batch(std::slice::from_ref(&words)).unwrap().remove(0);
        let (keep, ins) = tagger.tag(&v, 0.5).unwrap();
        assert_eq!(apply(&words, &keep, &ins), t("the food is really good ."));
        let bad = t("the weather is bad .");
        let v = bert.contextual_batch(std::slice::from_ref(&bad)).unwrap().remove(0);
        let (keep, ins) = tagger.tag(&v, 0.5).unwrap();
        assert_eq!(apply(&bad, &keep, &ins), bad, "no insertion where none belongs");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_edit_the_tagger_would_make_ranks_above_others() {
        let ex = |very_at: usize| {
            let vectors: Vec<Vec<f32>> = (0..4).map(|i| if i == very_at { vec![1.0, 0.0] } else { vec![0.0, 1.0] }).collect();
            Example { keep: (0..4).map(|i| i != very_at).collect(), inserts: vec![0; 5], vectors }
        };
        let data: Vec<Example> = (0..40).map(|i| ex(i % 4)).collect();
        let tagger = EditTagger::train(&data, vec!["really".into()], 2, 0, 60, 0.05, 7).unwrap();
        let before = t("a very b c");
        let likely = tagger.edit_log_likelihood(&data[1].vectors, &before, &t("a b c")).unwrap();
        let unlikely = tagger.edit_log_likelihood(&data[1].vectors, &before, &t("very b c")).unwrap();
        let unchanged = tagger.edit_log_likelihood(&data[1].vectors, &before, &before).unwrap();
        assert!(likely > unchanged && unchanged > unlikely, "{likely} > {unchanged} > {unlikely}");
    }

    #[test]
    fn negations_numbers_and_names_are_protected() {
        for (i, w) in [(3, "didnt"), (2, "not"), (1, "n't"), (4, "1986"), (5, "Romania")] { assert!(protected(i, w), "{w}") }
        for (i, w) in [(0, "The"), (2, "meeting"), (3, "interest")] { assert!(!protected(i, w), "{w}") }
    }

    #[test]
    fn writing_needs_an_inserted_or_replaced_word() {
        assert!(!writes(&t("It is endemic to the island"), &t("It is to the island")), "pure deletion");
        assert!(writes(&t("provide the information"), &t("provide details")), "replacement");
        assert!(writes(&t("a civil rather criminal matter"), &t("a civil rather than criminal matter")), "insertion");
        assert!(!writes(&t("a b ."), &t("a b , .")), "punctuation alone writes nothing");
    }

    #[test]
    fn rewording_excludes_name_and_number_edits() {
        assert!(rewords(&t("we did use it"), &t("we used it")));
        assert!(!rewords(&t("premier Gordon Campbell 's office"), &t("premier Campbell 's office")));
        assert!(!rewords(&t("about 300 people"), &t("about 400 people")));
        assert!(rewords(&t("The results is good"), &t("Results are good")), "the first word may be capitalized");
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
        let tagger = EditTagger::train(&data, vec!["really".into()], 2, 0, 60, 0.05, 7).unwrap();
        let (keep, inserts) = tagger.tag(&data[1].vectors, 0.5).unwrap();
        assert_eq!(keep, data[1].keep);
        assert!(inserts.iter().all(Vec::is_empty));
        let deep = EditTagger::train(&data, vec!["really".into()], 2, 8, 60, 0.05, 7).unwrap();
        assert_eq!(deep.tag(&data[1].vectors, 0.5).unwrap().0, data[1].keep, "with a hidden layer too");
        let dir = std::env::temp_dir().join(format!("edit-tagger-{}", std::process::id()));
        tagger.save(&dir).unwrap();
        let back = EditTagger::load(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(back.tag(&data[1].vectors, 0.5).unwrap().0, data[1].keep);
    }
}
