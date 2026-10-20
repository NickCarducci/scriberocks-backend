//! A small pre-LayerNorm transformer encoder with a tied output layer,
//! trained to fill in masked words — and, optionally, to name each masked
//! word's supersense (sense modeling, after SenseBERT; Levine et al. 2020).

use crate::usage::masking::{mask, Rng};
use crate::usage::morph::Morphology;
use crate::usage::vocab::{Vocab, PAD, UNK};
pub use candle_core::Device;
use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{linear, AdamW, Embedding, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Config {
    /// Filled in from the vocabulary.
    pub vocab_size: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub ff: usize,
    pub max_len: usize,
}

/// Sentences per forward pass in `contextual_batch`. Bounds peak memory:
/// at max_len 48 and ff 1024, one chunk's activations are a few hundred MB,
/// whereas tens of thousands of sentences at once can exhaust the machine.
pub const CONTEXT_CHUNK: usize = 128;

pub struct TrainOptions {
    /// Total steps in the schedule (a resumed run still counts from 0).
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
    pub warmup: usize,
    pub seed: u64,
    /// Where to start in the schedule (from `resume`).
    pub start_step: usize,
    /// Save weights + progress to this directory every N steps and at the end.
    pub checkpoint: Option<(std::path::PathBuf, usize)>,
}

/// LayerNorm from plain differentiable ops.
struct Norm {
    weight: Tensor,
    bias: Tensor,
}

impl Norm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Norm> {
        Ok(Norm {
            weight: vb.get_with_hints(dim, "weight", candle_nn::Init::Const(1.0))?,
            bias: vb.get_with_hints(dim, "bias", candle_nn::Init::Const(0.0))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        centered.broadcast_div(&(var + 1e-5)?.sqrt()?)?.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

struct Layer {
    ln1: Norm,
    qkv: Linear,
    out: Linear,
    ln2: Norm,
    ff1: Linear,
    ff2: Linear,
}

struct Encoder {
    tok: Embedding,
    pos: Embedding,
    layers: Vec<Layer>,
    ln_f: Norm,
    config: Config,
}

impl Encoder {
    fn new(c: Config, vb: VarBuilder) -> Result<Encoder> {
        let layers = (0..c.layers)
            .map(|i| {
                let vb = vb.pp(format!("layer{i}"));
                Ok(Layer {
                    ln1: Norm::new(c.dim, vb.pp("ln1"))?,
                    qkv: linear(c.dim, 3 * c.dim, vb.pp("qkv"))?,
                    out: linear(c.dim, c.dim, vb.pp("out"))?,
                    ln2: Norm::new(c.dim, vb.pp("ln2"))?,
                    ff1: linear(c.dim, c.ff, vb.pp("ff1"))?,
                    ff2: linear(c.ff, c.dim, vb.pp("ff2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // Small init (σ = 0.02): the output layer reuses `tok`, and σ = 1
        // makes the first logits enormous.
        let small = candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 };
        let tok = vb.pp("tok").get_with_hints((c.vocab_size, c.dim), "weight", small)?;
        let pos = vb.pp("pos").get_with_hints((c.max_len, c.dim), "weight", small)?;
        Ok(Encoder {
            tok: Embedding::new(tok, c.dim),
            pos: Embedding::new(pos, c.dim),
            layers,
            ln_f: Norm::new(c.dim, vb.pp("ln_f"))?,
            config: c,
        })
    }

    /// ids: [batch, len] u32 → hidden states [batch, len, dim].
    fn hidden(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, t) = ids.dims2()?;
        let (d, h) = (self.config.dim, self.config.heads);
        let dh = d / h;
        let positions = Tensor::arange(0u32, t as u32, ids.device())?;
        let mut x = self.tok.forward(ids)?.broadcast_add(&self.pos.forward(&positions)?)?;
        // Additive attention mask: padding keys get -1e9.
        let pad = ids.eq(PAD)?.to_dtype(DType::F32)?.affine(-1e9, 0.0)?.reshape((b, 1, 1, t))?;
        for l in &self.layers {
            let qkv = l.qkv.forward(&l.ln1.forward(&x)?)?.reshape((b, t, 3, h, dh))?;
            let q = qkv.narrow(2, 0, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
            let k = qkv.narrow(2, 1, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
            let v = qkv.narrow(2, 2, 1)?.squeeze(2)?.transpose(1, 2)?.contiguous()?;
            let att = (q.matmul(&k.t()?)? / (dh as f64).sqrt())?.broadcast_add(&pad)?;
            // Not `softmax_last_dim`: that fused op has no backward pass in
            // candle 0.11, which silently freezes attention during training.
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            let o = att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, d))?;
            x = (x + l.out.forward(&o)?)?;
            let f = l.ff2.forward(&l.ff1.forward(&l.ln2.forward(&x)?)?.gelu_erf()?)?;
            x = (x + f)?;
        }
        self.ln_f.forward(&x)
    }

    /// Logits over the vocabulary (output tied to the input embeddings).
    fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        hidden.broadcast_matmul(&self.tok.embeddings().t()?)
    }
}

pub struct UsageModel {
    pub vocab: Vocab,
    pub config: Config,
    varmap: VarMap,
    encoder: Encoder,
    device: Device,
    /// When set, words are split into base + ending pieces before encoding.
    morph: Option<Morphology>,
    /// Supersense table [categories, dim], when sense modeling is on.
    supersense: Option<Tensor>,
}

#[derive(Serialize, Deserialize)]
struct Meta {
    vocab: Vocab,
    config: Config,
    #[serde(default)]
    morph: Option<Morphology>,
    #[serde(default)]
    supersenses: usize,
}

/// Supersense targets for sense modeling. A masked word's target is its gold
/// supersense when the corpus has one (SemCor), else the set the dictionary
/// allows for that word; the loss rewards probability on any allowed one
/// (SenseBERT's weak supervision), so context decides between them.
pub struct SenseTargets {
    /// Allowed supersenses per vocabulary id (empty = no target).
    pub by_id: Vec<Vec<u16>>,
    /// Per sentence (same order as the training sentences), gold supersense
    /// per position; an empty Vec means none for that sentence.
    pub gold: Vec<Vec<Option<u16>>>,
    /// Weight of the supersense loss next to the masked-word loss.
    pub weight: f32,
}

impl UsageModel {
    pub fn new_random(vocab: Vocab, mut config: Config, device: Device, seed: u64) -> Result<UsageModel> {
        config.vocab_size = vocab.len();
        let _ = device.set_seed(seed); // not supported on CPU; init is random there
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let encoder = Encoder::new(config, vb)?;
        Ok(UsageModel { vocab, config, varmap, encoder, device, morph: None, supersense: None })
    }

    /// Adds a supersense table (sense modeling) with `n` categories.
    pub fn with_supersenses(mut self, n: usize) -> Result<UsageModel> {
        let vb = VarBuilder::from_varmap(&self.varmap, DType::F32, &self.device);
        let small = candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 };
        self.supersense = Some(vb.pp("supersense").get_with_hints((n, self.config.dim), "weight", small)?);
        Ok(self)
    }

    /// Number of supersense categories (0 when sense modeling is off).
    pub fn supersenses(&self) -> usize {
        self.supersense.as_ref().map_or(0, |t| t.dims()[0])
    }

    /// Log-probability of each supersense at every word (truncated to
    /// max_len); None without a supersense table. Pass "[MASK]" at a word
    /// to ask what its context alone says.
    pub fn supersense_log_probs(&self, words: &[String]) -> Result<Option<Vec<Vec<f32>>>> {
        let Some(table) = &self.supersense else { return Ok(None) };
        let (ids, base) = self.encode_window(words);
        let n = ids.len();
        let hidden = self.encoder.hidden(&Tensor::from_vec(ids, (1, n), &self.device)?)?.squeeze(0)?;
        let lp = candle_nn::ops::log_softmax(&hidden.matmul(&table.t()?)?, D::Minus1)?.to_vec2::<f32>()?;
        Ok(Some(base.into_iter().map(|b| lp[b].clone()).collect()))
    }

    pub fn with_morphology(mut self, morph: Morphology) -> UsageModel {
        self.morph = Some(morph);
        self
    }

    pub fn morphology(&self) -> Option<&Morphology> {
        self.morph.as_ref()
    }

    /// Pieces for `words` and each word's base-piece index.
    fn pieces(&self, words: &[String]) -> (Vec<String>, Vec<usize>) {
        match &self.morph {
            Some(m) => m.split_sentence(words),
            None => (words.to_vec(), (0..words.len()).collect()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Masked-word training. `log(step, loss)` is called every step.
    pub fn train(&mut self, sentences: &[Vec<u32>], opts: &TrainOptions, mut log: impl FnMut(usize, f32)) -> Result<()> {
        self.train_with_senses(sentences, opts, None, |step, loss, _| log(step, loss))
    }

    /// Masked-word training plus, with `senses` (and a supersense table),
    /// the supersense loss at masked positions. `log(step, word_loss,
    /// supersense_loss)` is called every step.
    pub fn train_with_senses(&mut self, sentences: &[Vec<u32>], opts: &TrainOptions, senses: Option<&SenseTargets>, mut log: impl FnMut(usize, f32, f32)) -> Result<()> {
        let usable: Vec<usize> = (0..sentences.len()).filter(|&i| sentences[i].len() >= 2).collect();
        if usable.is_empty() {
            return Ok(());
        }
        let mut opt = AdamW::new(self.varmap.all_vars(), ParamsAdamW { lr: opts.lr, weight_decay: 0.01, ..Default::default() })?;
        // A resumed run reseeds by step so it doesn't replay the same batches.
        let mut rng = Rng::new(opts.seed ^ (opts.start_step as u64).wrapping_mul(0x9E37));
        let max = self.config.max_len;
        for step in opts.start_step..opts.steps {
            // Linear warmup, then cosine decay to 10%.
            let lr = if step < opts.warmup {
                opts.lr * (step + 1) as f64 / opts.warmup as f64
            } else {
                let p = (step - opts.warmup) as f64 / (opts.steps - opts.warmup).max(1) as f64;
                opts.lr * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI * p).cos()))
            };
            opt.set_learning_rate(lr);

            let batch: Vec<usize> = (0..opts.batch).map(|_| usable[rng.below(usable.len() as u64) as usize]).collect();
            let len = batch.iter().map(|&s| sentences[s].len().min(max)).max().unwrap();
            let (mut input, mut flat_positions, mut targets) = (Vec::new(), Vec::new(), Vec::new());
            // Supersense targets: (index into the masked positions, allowed set).
            let mut sense_targets: Vec<(u32, &[u16])> = Vec::new();
            for (bi, &si) in batch.iter().enumerate() {
                let sent = &sentences[si];
                let start = if sent.len() > max { rng.below((sent.len() - max + 1) as u64) as usize } else { 0 };
                let window = &sent[start..(start + max).min(sent.len())];
                let m = mask(window, self.config.vocab_size as u32, &mut rng);
                for (i, label) in m.labels.iter().enumerate() {
                    if let Some(t) = label {
                        if let (Some(st), Some(_)) = (senses, &self.supersense) {
                            let gold = st.gold.get(si).and_then(|g| g.get(start + i)).copied().flatten();
                            let allowed: &[u16] = match gold {
                                Some(g) => std::slice::from_ref(&SUPERSENSE_IDS[(g as usize).min(63)]),
                                None => st.by_id.get(*t as usize).map_or(&[], Vec::as_slice),
                            };
                            if !allowed.is_empty() {
                                sense_targets.push((targets.len() as u32, allowed));
                            }
                        }
                        flat_positions.push((bi * len + i) as u32);
                        targets.push(*t);
                    }
                }
                input.extend(m.input.iter().copied());
                input.extend(std::iter::repeat_n(PAD, len - window.len()));
            }
            let ids = Tensor::from_vec(input, (opts.batch, len), &self.device)?;
            let hidden = self.encoder.hidden(&ids)?.reshape((opts.batch * len, self.config.dim))?;
            let picked = hidden.index_select(&Tensor::new(flat_positions.as_slice(), &self.device)?, 0)?;
            let logits = self.encoder.logits(&picked)?;
            let word_loss = candle_nn::loss::cross_entropy(&logits, &Tensor::new(targets.as_slice(), &self.device)?)?;
            let (loss, sense_loss) = match (&self.supersense, senses) {
                (Some(table), Some(st)) if !sense_targets.is_empty() => {
                    let c = table.dims()[0];
                    let rows: Vec<u32> = sense_targets.iter().map(|(r, _)| *r).collect();
                    let mut allowed = vec![0f32; rows.len() * c];
                    for (k, (_, set)) in sense_targets.iter().enumerate() {
                        for &s in *set {
                            if (s as usize) < c {
                                allowed[k * c + s as usize] = 1.0;
                            }
                        }
                    }
                    let h = picked.index_select(&Tensor::new(rows.as_slice(), &self.device)?, 0)?;
                    let p = candle_nn::ops::softmax(&h.matmul(&table.t()?)?, D::Minus1)?;
                    let mass = (p * Tensor::from_vec(allowed, (rows.len(), c), &self.device)?)?.sum(D::Minus1)?;
                    let sense_loss = ((mass + 1e-9)?.log()?.mean_all()? * -1.0)?;
                    ((&word_loss + (&sense_loss * st.weight as f64)?)?, sense_loss.to_scalar::<f32>()?)
                }
                _ => (word_loss.clone(), 0.0),
            };
            opt.backward_step(&loss)?;
            log(step, word_loss.to_scalar::<f32>()?, sense_loss);
            if let Some((dir, every)) = &opts.checkpoint {
                if (step + 1) % every == 0 || step + 1 == opts.steps {
                    self.checkpoint(dir, step + 1)?;
                }
            }
        }
        Ok(())
    }

    /// Saves weights, then records `step` — so a crash mid-save never
    /// claims progress whose weights aren't on disk.
    fn checkpoint(&self, dir: &Path, step: usize) -> Result<()> {
        self.save(dir)?;
        let tmp = dir.join("progress.json.tmp");
        std::fs::write(&tmp, format!("{{\"step\":{step}}}"))?;
        std::fs::rename(tmp, dir.join("progress.json"))?;
        Ok(())
    }

    /// Steps completed by the checkpoint in `dir`, if any.
    pub fn checkpoint_step(dir: &Path) -> Option<usize> {
        let text = std::fs::read_to_string(dir.join("progress.json")).ok()?;
        serde_json::from_str::<serde_json::Value>(&text).ok()?["step"].as_u64().map(|s| s as usize)
    }

    /// Loads a checkpoint and the step to continue from.
    pub fn resume(dir: &Path, device: Device) -> Result<(UsageModel, usize)> {
        let step = Self::checkpoint_step(dir).unwrap_or(0);
        Ok((Self::load(dir, device)?, step))
    }

    /// Encoded pieces (truncated to max_len) and the base piece of each word
    /// that fits.
    // MARK: fine-tuning a task head (tagging, parsing, senses) on the encoder

    /// Piece ids for `words` (truncated to max_len) and each word's base-piece index.
    pub fn encode_words(&self, words: &[String]) -> (Vec<u32>, Vec<usize>) {
        self.encode_window(words)
    }

    /// Hidden states [batch, len, dim] for padded piece ids (PAD = 0); differentiable,
    /// so a task head can be trained together with the encoder.
    pub fn hidden_states(&self, ids: &Tensor) -> Result<Tensor> {
        self.encoder.hidden(ids)
    }

    /// The encoder's parameters, for an optimizer that fine-tunes them.
    pub fn trainable_vars(&self) -> Vec<candle_core::Var> {
        self.varmap.all_vars()
    }

    fn encode_window(&self, words: &[String]) -> (Vec<u32>, Vec<usize>) {
        let (pieces, base) = self.pieces(words);
        let ids: Vec<u32> = self.vocab.encode(&pieces).into_iter().take(self.config.max_len).collect();
        let base = base.into_iter().take_while(|&b| b < ids.len()).collect();
        (ids, base)
    }

    /// Log-probability of each candidate filling `position` in `words`.
    /// A multiword candidate (or one split into base + ending) fills that
    /// many masked slots and scores the sum — the whole phrase's probability;
    /// averaging would let phrases of easy filler words ("all in") look like
    /// good fits. An unknown piece scores as [UNK] with a penalty.
    pub fn candidate_scores(&self, words: &[String], position: usize, candidates: &[String]) -> Result<Vec<f32>> {
        let before = self.pieces(&words[..position.min(words.len())]).0;
        let after = self.pieces(&words[(position + 1).min(words.len())..]).0;
        let mut out = Vec::with_capacity(candidates.len());
        let mut cache: std::collections::HashMap<usize, Vec<Vec<f32>>> = std::collections::HashMap::new();
        for cand in candidates {
            let parts: Vec<String> = self.pieces(&cand.split(' ').map(String::from).collect::<Vec<_>>()).0;
            let k = parts.len();
            if before.len() + k > self.config.max_len {
                out.push(f32::NEG_INFINITY);
                continue;
            }
            if !cache.contains_key(&k) {
                let mut seq = before.clone();
                seq.extend(std::iter::repeat_n("[MASK]".to_string(), k));
                seq.extend(after.iter().cloned());
                let ids: Vec<u32> = self.vocab.encode(&seq).into_iter().take(self.config.max_len).collect();
                let n = ids.len();
                let t = Tensor::from_vec(ids, (1, n), &self.device)?;
                let hidden = self.encoder.hidden(&t)?.squeeze(0)?;
                let slots = hidden.narrow(0, before.len(), k)?;
                let lp = candle_nn::ops::log_softmax(&self.encoder.logits(&slots)?, D::Minus1)?.to_vec2::<f32>()?;
                cache.insert(k, lp);
            }
            let log_probs = &cache[&k];
            let mut total = 0.0;
            for (i, part) in parts.iter().enumerate() {
                let id = self.vocab.id(part);
                total += log_probs[i][id as usize] - if id == UNK { 5.0 } else { 0.0 };
            }
            out.push(total);
        }
        Ok(out)
    }

    /// Contextual vector for every word (truncated to max_len).
    pub fn contextual(&self, words: &[String]) -> Result<Vec<Vec<f32>>> {
        let (ids, base) = self.encode_window(words);
        let n = ids.len();
        let t = Tensor::from_vec(ids, (1, n), &self.device)?;
        let hidden = self.encoder.hidden(&t)?.squeeze(0)?.to_vec2::<f32>()?;
        Ok(base.into_iter().map(|b| hidden[b].clone()).collect())
    }

    /// `contextual` for many sentences in one pass (padded; padding is masked
    /// out of attention, so results match one-at-a-time).
    pub fn contextual_batch(&self, sentences: &[Vec<String>]) -> Result<Vec<Vec<Vec<f32>>>> {
        if sentences.len() > CONTEXT_CHUNK {
            let mut out = Vec::with_capacity(sentences.len());
            for chunk in sentences.chunks(CONTEXT_CHUNK) {
                out.extend(self.contextual_batch(chunk)?);
            }
            return Ok(out);
        }
        if sentences.is_empty() {
            return Ok(vec![]);
        }
        let encoded: Vec<(Vec<u32>, Vec<usize>)> = sentences.iter().map(|s| self.encode_window(s)).collect();
        let len = encoded.iter().map(|(ids, _)| ids.len()).max().unwrap_or(1).max(1);
        let mut flat = Vec::with_capacity(encoded.len() * len);
        for (ids, _) in &encoded {
            flat.extend(ids.iter().copied());
            flat.extend(std::iter::repeat_n(PAD, len - ids.len()));
        }
        let t = Tensor::from_vec(flat, (encoded.len(), len), &self.device)?;
        let hidden = self.encoder.hidden(&t)?.to_vec3::<f32>()?;
        Ok(hidden.into_iter().zip(&encoded).map(|(h, (_, base))| base.iter().map(|&b| h[b].clone()).collect()).collect())
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        self.varmap.save(dir.join("usage.safetensors"))?;
        let meta = serde_json::to_vec(&Meta { vocab: self.vocab.clone(), config: self.config, morph: self.morph.clone(), supersenses: self.supersenses() })
            .map_err(candle_core::Error::wrap)?;
        std::fs::write(dir.join("usage.json"), meta)?;
        Ok(())
    }

    pub fn load(dir: &Path, device: Device) -> Result<UsageModel> {
        let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("usage.json"))?).map_err(candle_core::Error::wrap)?;
        let mut vocab = meta.vocab;
        vocab.reindex();
        let mut model = UsageModel::new_random(vocab, meta.config, device, 0)?;
        if meta.supersenses > 0 {
            model = model.with_supersenses(meta.supersenses)?;
        }
        model.morph = meta.morph;
        model.varmap.load(dir.join("usage.safetensors"))?;
        Ok(model)
    }
}
/// Supersense ids 0..64, so a gold supersense can be lent as a one-element
/// allowed set (WordNet has 45).
const SUPERSENSE_IDS: [u16; 64] = {
    let mut a = [0u16; 64];
    let mut i = 0;
    while i < 64 {
        a[i] = i as u16;
        i += 1;
    }
    a
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usage::vocab::Vocab;

    fn corpus() -> Vec<Vec<String>> {
        let lines = [
            "the cat sat on the mat", "the cat slept on the mat", "a dog sat on a rug", "a dog slept on a rug",
            "the cat ate the fish", "a dog ate a bone", "the bird sang in the tree", "a bird sat in a tree",
        ];
        lines.iter().cycle().take(64).map(|l| l.split(' ').map(String::from).collect()).collect()
    }

    fn tiny() -> Config {
        Config { vocab_size: 0, dim: 32, layers: 2, heads: 2, ff: 64, max_len: 16 }
    }

    fn trained() -> (UsageModel, f32, f32) {
        let sents = corpus();
        let vocab = Vocab::build(&sents, 1, 100);
        let mut model = UsageModel::new_random(vocab, tiny(), Device::Cpu, 1).unwrap();
        let ids: Vec<Vec<u32>> = sents.iter().map(|s| model.vocab.encode(s)).collect();
        let mut losses = Vec::new();
        model
            .train(&ids, &TrainOptions { steps: 1200, batch: 16, lr: 3e-3, warmup: 50, seed: 1, start_step: 0, checkpoint: None }, |_, loss| losses.push(loss))
            .unwrap();
        let first = losses[..10].iter().sum::<f32>() / 10.0;
        let last = losses[losses.len() - 10..].iter().sum::<f32>() / 10.0;
        (model, first, last)
    }

    fn words(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    #[test]
    fn learns_to_fill_in_words_from_context() {
        let (model, first, last) = trained();
        assert!(last < first * 0.6, "loss {first} → {last}");
        let cands = words("mat rug fish");
        let scores = model.candidate_scores(&words("the cat sat on the mat"), 5, &cands).unwrap();
        assert!(scores[0] > scores[1] && scores[0] > scores[2], "{scores:?}");
        let scores = model.candidate_scores(&words("a dog sat on a rug"), 5, &cands).unwrap();
        assert!(scores[1] > scores[0], "{scores:?}");
    }

    #[test]
    fn a_phrase_of_common_filler_words_does_not_outscore_the_right_word() {
        let (model, _, _) = trained();
        let c = vec!["mat".to_string(), "the on".to_string(), "on the a".to_string()];
        let scores = model.candidate_scores(&words("the cat sat on the mat"), 5, &c).unwrap();
        assert!(scores[0] > scores[1] && scores[0] > scores[2], "{scores:?}");
    }

    #[test]
    fn save_and_load_give_identical_scores() {
        let (model, _, _) = trained();
        let dir = std::env::temp_dir().join(format!("usage-{}", std::process::id()));
        model.save(&dir).unwrap();
        let loaded = UsageModel::load(&dir, Device::Cpu).unwrap();
        let c = words("mat rug");
        let s = words("the cat sat on the mat");
        assert_eq!(model.candidate_scores(&s, 5, &c).unwrap(), loaded.candidate_scores(&s, 5, &c).unwrap());
    }

    #[test]
    fn scores_multiword_and_unknown_candidates() {
        let (model, _, _) = trained();
        let scores = model.candidate_scores(&words("the cat sat on the mat"), 5, &words("mat").into_iter().chain(["the mat".into(), "zebra".into()]).collect::<Vec<_>>()).unwrap();
        assert!(scores.iter().all(|s| s.is_finite()), "{scores:?}");
        assert!(scores[0] > scores[2], "known fits better than unknown: {scores:?}");
    }

    #[test]
    fn training_checkpoints_and_resumes_where_it_left_off() {
        let sents = corpus();
        let vocab = Vocab::build(&sents, 1, 100);
        let ids: Vec<Vec<u32>> = sents.iter().map(|s| vocab.encode(s)).collect();
        let dir = std::env::temp_dir().join(format!("usage-ckpt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        // "Interrupted" run: 40 of 60 steps, checkpoint every 20.
        let mut model = UsageModel::new_random(vocab, tiny(), Device::Cpu, 1).unwrap();
        let opts = |steps| TrainOptions { steps, batch: 8, lr: 3e-3, warmup: 5, seed: 1, start_step: 0, checkpoint: Some((dir.clone(), 20)) };
        model.train(&ids, &opts(40), |_, _| {}).unwrap();
        assert_eq!(UsageModel::checkpoint_step(&dir), Some(40));

        let (mut resumed, step) = UsageModel::resume(&dir, Device::Cpu).unwrap();
        assert_eq!(step, 40);
        let mut seen = vec![];
        resumed.train(&ids, &TrainOptions { start_step: step, ..opts(60) }, |s, _| seen.push(s)).unwrap();
        assert_eq!((seen.first().copied(), seen.last().copied(), seen.len()), (Some(40), Some(59), 20));
        assert_eq!(UsageModel::checkpoint_step(&dir), Some(60));
    }

    #[test]
    fn with_morphology_words_are_split_but_callers_still_see_words() {
        use crate::usage::morph::Morphology;
        let lemmas = |w: &str| -> Vec<String> {
            match w { "cats" => vec!["cat".into()], "dogs" => vec!["dog".into()], "sat" => vec!["sit".into()], o => vec![o.into()] }
        };
        let lines = ["the cats sat on the mat", "the dogs sat on the rug", "a cat sat on a mat", "a dog sat on a rug"];
        let sents: Vec<Vec<String>> = lines.iter().cycle().take(64).map(|l| words(l)).collect();
        let morph = Morphology::build(sents.iter().flatten().cloned(), lemmas);
        let pieces: Vec<Vec<String>> = sents.iter().map(|s| morph.split_sentence(s).0).collect();
        let vocab = Vocab::build(&pieces, 1, 100);
        assert!(vocab.contains("##s") && vocab.contains("cat") && !vocab.contains("cats"));
        let mut model = UsageModel::new_random(vocab, tiny(), Device::Cpu, 1).unwrap().with_morphology(morph);
        let ids: Vec<Vec<u32>> = pieces.iter().map(|s| model.vocab.encode(s)).collect();
        model.train(&ids, &TrainOptions { steps: 600, batch: 16, lr: 3e-3, warmup: 30, seed: 1, start_step: 0, checkpoint: None }, |_, _| {}).unwrap();

        let sentence = words("the cats sat on the mat");
        assert_eq!(model.contextual(&sentence).unwrap().len(), 6, "one vector per word");
        let scores = model.candidate_scores(&sentence, 1, &words("cats dogs zebras")).unwrap();
        assert!(scores.iter().all(|x| x.is_finite()) && scores[0] > scores[2], "{scores:?}");

        let dir = std::env::temp_dir().join(format!("usage-morph-{}", std::process::id()));
        model.save(&dir).unwrap();
        let loaded = UsageModel::load(&dir, Device::Cpu).unwrap();
        assert_eq!(loaded.candidate_scores(&sentence, 1, &words("cats dogs")).unwrap(), model.candidate_scores(&sentence, 1, &words("cats dogs")).unwrap());
    }

    #[test]
    fn words_past_the_window_get_no_vector_when_pieces_overflow() {
        use crate::usage::morph::Morphology;
        let lemmas = |w: &str| -> Vec<String> { if w == "cats" { vec!["cat".into()] } else { vec![w.into()] } };
        let morph = Morphology::build(["cats".to_string()].into_iter(), lemmas);
        let vocab = Vocab::build(&[words("cat ##s the")], 1, 100);
        let model = UsageModel::new_random(vocab, tiny(), Device::Cpu, 1).unwrap().with_morphology(morph);
        // 10 words → 20 pieces; window is 16 pieces, so only 8 words fit.
        let sentence: Vec<String> = std::iter::repeat_n("cats".to_string(), 10).collect();
        assert_eq!(model.contextual(&sentence).unwrap().len(), 8);
        assert_eq!(model.contextual_batch(&[sentence]).unwrap()[0].len(), 8);
    }

    #[test]
    fn large_batches_are_processed_in_bounded_chunks() {
        // Thousands of sentences in one pass exhausted a 48 GB Mac; the method
        // must chunk internally and still return every sentence's vectors.
        let (model, _, _) = trained();
        let n = 3 * CONTEXT_CHUNK + 5;
        let sents: Vec<Vec<String>> = (0..n).map(|i| words(if i % 2 == 0 { "the cat sat" } else { "a dog slept on a rug" })).collect();
        let out = model.contextual_batch(&sents).unwrap();
        assert_eq!(out.len(), n);
        let last = model.contextual(&sents[n - 1]).unwrap();
        for (x, y) in last.iter().flatten().zip(out[n - 1].iter().flatten()) {
            assert!((x - y).abs() < 1e-4);
        }
    }

    #[test]
    fn batched_contextual_matches_one_at_a_time() {
        let (model, _, _) = trained();
        let sents: Vec<Vec<String>> = ["the cat sat", "a dog slept on a rug", "bird"].iter().map(|s| words(s)).collect();
        let batched = model.contextual_batch(&sents).unwrap();
        for (s, b) in sents.iter().zip(&batched) {
            let single = model.contextual(s).unwrap();
            assert_eq!(single.len(), b.len());
            for (x, y) in single.iter().flatten().zip(b.iter().flatten()) {
                assert!((x - y).abs() < 1e-4, "{x} vs {y}");
            }
        }
    }

    #[test]
    fn contextual_vectors_have_model_width() {
        let (model, _, _) = trained();
        let v = model.contextual(&words("the cat sat")).unwrap();
        assert_eq!((v.len(), v[0].len()), (3, 32));
    }
}

#[cfg(test)]
mod debug {
    use super::*;
    #[test]
    #[ignore]
    fn show_predictions() {
        let lines = ["the cat sat on the mat", "the cat slept on the mat", "a dog sat on a rug", "a dog slept on a rug", "the cat ate the fish", "a dog ate a bone", "the bird sang in the tree", "a bird sat in a tree"];
        let sents: Vec<Vec<String>> = lines.iter().cycle().take(64).map(|l| l.split(' ').map(String::from).collect()).collect();
        let vocab = Vocab::build(&sents, 1, 100);
        let mut m = UsageModel::new_random(vocab, Config { vocab_size: 0, dim: 32, layers: 2, heads: 2, ff: 64, max_len: 16 }, Device::Cpu, 1).unwrap();
        let ids: Vec<Vec<u32>> = sents.iter().map(|s| m.vocab.encode(s)).collect();
        let mut losses = vec![];
        m.train(&ids, &TrainOptions { steps: 400, batch: 16, lr: 3e-3, warmup: 20, seed: 1, start_step: 0, checkpoint: None }, |_, l| losses.push(l)).unwrap();
        println!("loss first {:?} last {:?}", &losses[..3], &losses[397..]);
        let all: Vec<String> = (4..m.vocab.len() as u32).map(|i| m.vocab.word(i).to_string()).collect();
        for (s, p) in [("the cat sat on the mat", 5), ("a dog sat on a rug", 5), ("the cat sat on the mat", 1)] {
            let w: Vec<String> = s.split(' ').map(String::from).collect();
            let sc = m.candidate_scores(&w, p, &all).unwrap();
            let mut r: Vec<_> = all.iter().zip(sc).collect();
            r.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!("{s} @{p}: {:?}", &r[..4]);
        }
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    use crate::usage::masking::Rng;

    /// Second half repeats the first, so masked words are recoverable from
    /// context alone. A working encoder must get far below chance (ln 20 ≈ 3.0).
    #[test]
    fn learns_a_copy_task_that_requires_attention() {
        let symbols: Vec<String> = (0..20).map(|i| format!("s{i}")).collect();
        let vocab = Vocab::from_words(crate::usage::vocab::SPECIALS.iter().map(|s| s.to_string()).chain(symbols.iter().cloned()).collect());
        let mut rng = Rng::new(9);
        let ids: Vec<Vec<u32>> = (0..2000)
            .map(|_| {
                let half: Vec<u32> = (0..5).map(|_| 4 + rng.below(20) as u32).collect();
                half.iter().chain(half.iter()).copied().collect()
            })
            .collect();
        let mut model = UsageModel::new_random(vocab, Config { vocab_size: 0, dim: 64, layers: 2, heads: 4, ff: 128, max_len: 16 }, Device::Cpu, 3).unwrap();
        let mut losses = vec![];
        model.train(&ids, &TrainOptions { steps: 1500, batch: 32, lr: 1e-3, warmup: 100, seed: 3, start_step: 0, checkpoint: None }, |_, l| losses.push(l)).unwrap();
        let last = losses[losses.len() - 50..].iter().sum::<f32>() / 50.0;
        assert!(last < 1.5, "copy-task loss {last}");
    }
}

#[cfg(test)]
mod gradient_tests {
    use super::*;
    use crate::usage::vocab::MASK;

    /// Every parameter must receive a gradient. (candle 0.11's fused
    /// LayerNorm and softmax have no backward pass and silently cut the graph.)
    #[test]
    fn every_parameter_gets_a_gradient() {
        let vocab = Vocab::from_words(crate::usage::vocab::SPECIALS.iter().map(|s| s.to_string()).chain((0..20).map(|i| format!("s{i}"))).collect());
        let m = UsageModel::new_random(vocab, Config { vocab_size: 0, dim: 32, layers: 1, heads: 2, ff: 64, max_len: 16 }, Device::Cpu, 3).unwrap();
        let ids = Tensor::new(&[[4u32, 5, MASK, 7, 4, 5, 6, 7]], &Device::Cpu).unwrap();
        let h = m.encoder.hidden(&ids).unwrap().squeeze(0).unwrap();
        let logits = m.encoder.logits(&h.narrow(0, 2, 1).unwrap()).unwrap();
        let loss = candle_nn::loss::cross_entropy(&logits, &Tensor::new(&[6u32], &Device::Cpu).unwrap()).unwrap();
        let grads = loss.backward().unwrap();
        let data = m.varmap.data().lock().unwrap();
        let missing: Vec<&String> = data.iter().filter(|(_, v)| grads.get(v.as_tensor()).is_none()).map(|(n, _)| n).collect();
        assert!(missing.is_empty(), "no gradient for {missing:?}");
    }
}

#[cfg(test)]
mod sense_modeling_tests {
    use super::*;
    use crate::usage::vocab::Vocab;

    fn words(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    /// "bank" is ambiguous (river 0 / money 1); context words carry the sense.
    fn setup() -> (UsageModel, Vec<Vec<u32>>, SenseTargets) {
        let lines = ["the river bank was wet", "the money bank was rich", "a river bank is muddy", "a money bank is safe"];
        let sents: Vec<Vec<String>> = lines.iter().cycle().take(64).map(|l| words(l)).collect();
        let vocab = Vocab::build(&sents, 1, 100);
        let model = UsageModel::new_random(vocab, Config { vocab_size: 0, dim: 32, layers: 2, heads: 2, ff: 64, max_len: 16 }, Device::Cpu, 1)
            .unwrap()
            .with_supersenses(3)
            .unwrap();
        let ids: Vec<Vec<u32>> = sents.iter().map(|s| model.vocab.encode(s)).collect();
        let mut by_id = vec![vec![]; model.vocab.len()];
        by_id[model.vocab.id("bank") as usize] = vec![0, 1];
        by_id[model.vocab.id("river") as usize] = vec![0];
        by_id[model.vocab.id("money") as usize] = vec![1];
        by_id[model.vocab.id("muddy") as usize] = vec![2];
        // Gold for "bank" in the first two lines of each cycle (SemCor-style).
        let gold = sents.iter().map(|s| s.iter().map(|w| match (w.as_str(), s[1].as_str()) { ("bank", "river") => Some(0), ("bank", "money") => Some(1), _ => None }).collect()).collect();
        (model, ids, SenseTargets { by_id, gold, weight: 1.0 })
    }

    fn opts() -> TrainOptions {
        TrainOptions { steps: 800, batch: 16, lr: 3e-3, warmup: 50, seed: 1, start_step: 0, checkpoint: None }
    }

    #[test]
    fn context_decides_the_supersense_of_an_ambiguous_word() {
        let (mut model, ids, targets) = setup();
        let mut sense_losses = vec![];
        model.train_with_senses(&ids, &opts(), Some(&targets), |_, _, s| sense_losses.push(s)).unwrap();
        let river = model.supersense_log_probs(&words("the river [MASK] was wet")).unwrap().unwrap();
        let money = model.supersense_log_probs(&words("the money [MASK] was rich")).unwrap().unwrap();
        assert!(river[2][0] > river[2][1], "{:?}", river[2]);
        assert!(money[2][1] > money[2][0], "{:?}", money[2]);
        assert!(sense_losses.iter().any(|&l| l > 0.0), "supersense loss was applied");
    }

    #[test]
    fn a_word_with_one_allowed_supersense_learns_it_without_gold() {
        let (mut model, ids, mut targets) = setup();
        targets.gold.clear();
        model.train_with_senses(&ids, &opts(), Some(&targets), |_, _, _| {}).unwrap();
        let lp = model.supersense_log_probs(&words("a river bank is muddy")).unwrap().unwrap();
        let best = (0..3).max_by(|&a, &b| lp[4][a].total_cmp(&lp[4][b])).unwrap();
        assert_eq!(best, 2, "muddy → its only supersense: {:?}", lp[4]);
    }

    #[test]
    fn the_supersense_table_is_saved_and_loaded_and_old_models_have_none() {
        let (mut model, ids, targets) = setup();
        model.train_with_senses(&ids, &TrainOptions { steps: 20, ..opts() }, Some(&targets), |_, _, _| {}).unwrap();
        let dir = std::env::temp_dir().join(format!("usage-senses-{}", std::process::id()));
        model.save(&dir).unwrap();
        let loaded = UsageModel::load(&dir, Device::Cpu).unwrap();
        assert_eq!(loaded.supersenses(), 3);
        let s = words("the river bank was wet");
        assert_eq!(model.supersense_log_probs(&s).unwrap(), loaded.supersense_log_probs(&s).unwrap());
        let plain = UsageModel::new_random(model.vocab.clone(), model.config, Device::Cpu, 1).unwrap();
        assert_eq!(plain.supersenses(), 0);
        assert!(plain.supersense_log_probs(&s).unwrap().is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
