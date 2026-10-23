//! A BERT encoder with its masked-word head, in candle, for comparison with
//! the usage model on the same tests (and on the same runtime the app uses).
//! Loads a Hugging Face BERT checkpoint converted to safetensors (names as in
//! BertForPreTraining: `bert.*`, `cls.predictions.*`) plus its `vocab.txt`
//! and `config.json`. Uncased WordPiece; inputs are already-tokenized words.
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, linear, Embedding, Linear, VarBuilder, VarMap};

/// LayerNorm written out from differentiable ops. candle_nn's LayerNorm runs a
/// fused kernel with no backward pass, so gradients silently stop at it.
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

fn layer_norm(dim: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    Ok(LayerNorm { weight: vb.get(dim, "weight")?, bias: vb.get(dim, "bias")?, eps })
}

impl LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        centered.broadcast_div(&(var + self.eps)?.sqrt()?)?.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}
use std::collections::HashMap;
use std::path::Path;

/// 8-bit storage for shipping: every matrix row as bytes (value + 128) with
/// one f32 scale (largest magnitude / 127); vectors (biases, norms) stay f32.
/// About a quarter of the full-precision size; decompressed once on load.
pub fn quantize_q8(weights: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    for (name, t) in weights {
        let t = t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        if t.rank() != 2 {
            out.insert(name.clone(), t);
            continue;
        }
        let (rows, cols) = t.dims2()?;
        let v = t.to_vec2::<f32>()?;
        let mut bytes = Vec::with_capacity(rows * cols);
        let mut scales = Vec::with_capacity(rows);
        for row in &v {
            let scale = row.iter().fold(0f32, |m, x| m.max(x.abs())) / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            bytes.extend(row.iter().map(|x| ((x * inv).round().clamp(-127.0, 127.0) as i32 + 128) as u8));
            scales.push(scale);
        }
        out.insert(format!("{name}.q8"), Tensor::from_vec(bytes, (rows, cols), &Device::Cpu)?);
        out.insert(format!("{name}.scale"), Tensor::from_vec(scales, (rows, 1), &Device::Cpu)?);
    }
    Ok(out)
}

/// Inverse of `quantize_q8`, to f32 tensors on `device`.
pub fn dequantize_q8(stored: &HashMap<String, Tensor>, device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    for (name, t) in stored {
        if let Some(base) = name.strip_suffix(".q8") {
            let scale = &stored[&format!("{base}.scale")];
            let w = (t.to_dtype(DType::F32)? - 128.0)?.broadcast_mul(scale)?;
            out.insert(base.to_string(), w.to_device(device)?);
        } else if !name.ends_with(".scale") {
            out.insert(name.clone(), t.to_dtype(DType::F32)?.to_device(device)?);
        }
    }
    Ok(out)
}

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    attn_out: Linear,
    attn_norm: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
    ff_norm: LayerNorm,
}

pub struct Bert {
    words: Embedding,
    positions: Embedding,
    types: Embedding,
    norm: LayerNorm,
    layers: Vec<Layer>,
    head_dense: Linear,
    head_norm: LayerNorm,
    decoder: Linear,
    heads: usize,
    vocab: Vec<String>,
    index: HashMap<String, u32>,
    max_len: usize,
    /// Pre-LayerNorm layout (normalize each sublayer's input; one final norm):
    /// trains reliably from scratch. BERT checkpoints are post-LN (false).
    pre_ln: Option<LayerNorm>,
    device: Device,
}

const CLS: &str = "[CLS]";
const SEP: &str = "[SEP]";
const MASK: &str = "[MASK]";
const UNK: &str = "[UNK]";

impl Bert {
    /// Loads `model.safetensors` (full or half precision), or else
    /// `model.q8.safetensors` (8-bit, see `quantize_q8`), decompressed to f32.
    pub fn load(dir: &Path, device: Device) -> std::result::Result<Bert, Box<dyn std::error::Error>> {
        if !dir.join("model.safetensors").exists() && dir.join("model.q8.safetensors").exists() {
            let weights = dequantize_q8(&candle_core::safetensors::load(dir.join("model.q8.safetensors"), &Device::Cpu)?, &device)?;
            return Bert::build(dir, VarBuilder::from_tensors(weights, DType::F32, &device), device);
        }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device)? };
        Bert::build(dir, vb, device)
    }

    /// As `load`, but with every weight in a VarMap so it can be trained
    /// (continued masked-word training); save with `VarMap::save`.
    pub fn load_trainable(dir: &Path, device: Device) -> std::result::Result<(Bert, VarMap), Box<dyn std::error::Error>> {
        let mut vars = VarMap::new();
        let bert = Bert::build(dir, VarBuilder::from_varmap(&vars, DType::F32, &device), device)?;
        vars.load(dir.join("model.safetensors"))?;
        Ok((bert, vars))
    }

    /// A fresh, randomly initialized model with `dir`'s config.json and
    /// vocab.txt (training from scratch); weights in the returned VarMap.
    /// Initialized as BERT is (weights normal with σ = 0.02, biases 0,
    /// LayerNorm scale 1): with the library's defaults (σ = 1 embeddings) a
    /// from-scratch model stalls at word frequencies.
    pub fn new_random(dir: &Path, device: Device) -> std::result::Result<(Bert, VarMap), Box<dyn std::error::Error>> {
        let vars = VarMap::new();
        let bert = Bert::build(dir, VarBuilder::from_varmap(&vars, DType::F32, &device), device.clone())?;
        for (name, var) in vars.data().lock().unwrap().iter() {
            let shape = var.shape().clone();
            let value = if name.contains("LayerNorm") || name.contains("final_norm") {
                if name.ends_with("bias") { Tensor::zeros(shape, DType::F32, &device)? } else { Tensor::ones(shape, DType::F32, &device)? }
            } else if name.ends_with("bias") {
                Tensor::zeros(shape, DType::F32, &device)?
            } else {
                Tensor::randn(0f32, 0.02, shape, &device)?
            };
            var.set(&value)?;
        }
        Ok((bert, vars))
    }

    fn build(dir: &Path, vb: VarBuilder, device: Device) -> std::result::Result<Bert, Box<dyn std::error::Error>> {
        let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let n = |k: &str| config[k].as_u64().unwrap_or(0) as usize;
        let (dim, layers, heads, ff, max_len) = (n("hidden_size"), n("num_hidden_layers"), n("num_attention_heads"), n("intermediate_size"), n("max_position_embeddings"));
        let eps = config["layer_norm_eps"].as_f64().unwrap_or(1e-12);
        let vocab: Vec<String> = std::fs::read_to_string(dir.join("vocab.txt"))?.lines().map(String::from).collect();
        let index = vocab.iter().enumerate().map(|(i, w)| (w.clone(), i as u32)).collect();
        let b = vb.pp("bert");
        let e = b.pp("embeddings");
        let layer = |i: usize| -> Result<Layer> {
            let l = b.pp(format!("encoder.layer.{i}"));
            Ok(Layer {
                q: linear(dim, dim, l.pp("attention.self.query"))?,
                k: linear(dim, dim, l.pp("attention.self.key"))?,
                v: linear(dim, dim, l.pp("attention.self.value"))?,
                attn_out: linear(dim, dim, l.pp("attention.output.dense"))?,
                attn_norm: layer_norm(dim, eps, l.pp("attention.output.LayerNorm"))?,
                ff_in: linear(dim, ff, l.pp("intermediate.dense"))?,
                ff_out: linear(ff, dim, l.pp("output.dense"))?,
                ff_norm: layer_norm(dim, eps, l.pp("output.LayerNorm"))?,
            })
        };
        let p = vb.pp("cls.predictions");
        Ok(Bert {
            words: embedding(vocab.len(), dim, e.pp("word_embeddings"))?,
            positions: embedding(max_len, dim, e.pp("position_embeddings"))?,
            types: embedding(2, dim, e.pp("token_type_embeddings"))?,
            norm: layer_norm(dim, eps, e.pp("LayerNorm"))?,
            layers: (0..layers).map(layer).collect::<Result<_>>()?,
            pre_ln: match config["pre_layer_norm"].as_bool().unwrap_or(false) {
                true => Some(layer_norm(dim, eps, b.pp("encoder.final_norm"))?),
                false => None,
            },
            head_dense: linear(dim, dim, p.pp("transform.dense"))?,
            head_norm: layer_norm(dim, eps, p.pp("transform.LayerNorm"))?,
            decoder: linear(dim, vocab.len(), p.pp("decoder"))?,
            heads,
            vocab,
            index,
            max_len,
            device,
        })
    }

    /// WordPiece pieces of one (already split) word, lowercased; [UNK] if it can't be spelled.
    pub fn word_pieces(&self, word: &str) -> Vec<String> {
        if word == MASK {
            return vec![MASK.into()];
        }
        let chars: Vec<char> = word.to_lowercase().chars().collect();
        let (mut out, mut start) = (Vec::new(), 0);
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while end > start {
                let piece: String = chars[start..end].iter().collect();
                let piece = if start > 0 { format!("##{piece}") } else { piece };
                if self.index.contains_key(&piece) {
                    found = Some(piece);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(p) => { out.push(p); start = end }
                None => return vec![UNK.into()],
            }
        }
        out
    }

    /// Piece ids for [CLS] words [SEP], and the index of each word's first piece.
    fn encode(&self, words: &[String]) -> (Vec<u32>, Vec<usize>) {
        let mut ids = vec![self.index[CLS]];
        let mut first = Vec::with_capacity(words.len());
        for w in words {
            first.push(ids.len());
            ids.extend(self.word_pieces(w).iter().map(|p| self.index.get(p).copied().unwrap_or(self.index[UNK])));
        }
        ids.push(self.index[SEP]);
        ids.truncate(self.max_len);
        (ids, first)
    }

    /// Hidden states [b, t, dim] for padded id rows (padding masked out of attention).
    fn hidden(&self, rows: &[Vec<u32>]) -> Result<Tensor> {
        let (b, t) = (rows.len(), rows.iter().map(Vec::len).max().unwrap_or(1));
        let mut flat = Vec::with_capacity(b * t);
        let mut pad = Vec::with_capacity(b * t);
        for r in rows {
            flat.extend(r.iter().copied());
            flat.extend(std::iter::repeat_n(0u32, t - r.len()));
            pad.extend(std::iter::repeat_n(0f32, r.len()));
            pad.extend(std::iter::repeat_n(-1e9f32, t - r.len()));
        }
        let ids = Tensor::from_vec(flat, (b, t), &self.device)?;
        let pos = Tensor::arange(0u32, t as u32, &self.device)?.unsqueeze(0)?.broadcast_as((b, t))?.contiguous()?;
        let mut x = (self.words.forward(&ids)? + self.positions.forward(&pos)?)?
            .broadcast_add(&self.types.forward(&Tensor::zeros((b, t), DType::U32, &self.device)?)?)?;
        x = self.norm.forward(&x)?;
        let mask = Tensor::from_vec(pad, (b, 1, 1, t), &self.device)?;
        let dim = x.dim(2)?;
        let dh = dim / self.heads;
        for l in &self.layers {
            let split = |y: Tensor| -> Result<Tensor> { y.reshape((b, t, self.heads, dh))?.transpose(1, 2)?.contiguous() };
            let pre = self.pre_ln.is_some();
            let a = if pre { l.attn_norm.forward(&x)? } else { x.clone() };
            let (q, k, v) = (split(l.q.forward(&a)?)?, split(l.k.forward(&a)?)?, split(l.v.forward(&a)?)?);
            let att = (q.matmul(&k.t()?)? / (dh as f64).sqrt())?.broadcast_add(&mask)?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?; // not softmax_last_dim: its fused kernel has no backward pass
            let y = l.attn_out.forward(&att.matmul(&v)?.transpose(1, 2)?.reshape((b, t, dim))?)?;
            x = if pre { (y + &x)? } else { l.attn_norm.forward(&(y + &x)?)? };
            let f = if pre { l.ff_norm.forward(&x)? } else { x.clone() };
            let h = l.ff_out.forward(&l.ff_in.forward(&f)?.gelu_erf()?)?;
            x = if pre { (h + &x)? } else { l.ff_norm.forward(&(h + &x)?)? };
        }
        match &self.pre_ln {
            Some(norm) => norm.forward(&x),
            None => Ok(x),
        }
    }

    /// Piece ids of a sentence with [CLS]/[SEP], for training.
    pub fn encode_sentence(&self, words: &[String]) -> Vec<u32> {
        self.encode(words).0
    }

    /// Piece ids and the index of each word's first piece.
    pub fn encode_words(&self, words: &[String]) -> (Vec<u32>, Vec<usize>) {
        self.encode(words)
    }

    /// The id of a word that is a single vocabulary entry, if it is one.
    pub fn single_piece(&self, word: &str) -> Option<u32> {
        self.index.get(&word.to_lowercase()).copied()
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    /// Log-probabilities over the vocabulary at `positions` (row, column)
    /// of the padded batch `rows`: [n, vocab]. The training objective.
    pub fn masked_log_probs(&self, rows: &[Vec<u32>], positions: &[(usize, usize)]) -> Result<Tensor> {
        let h = self.hidden(rows)?;
        let (b, t, dim) = h.dims3()?;
        let flat: Vec<u32> = positions.iter().map(|&(r, c)| (r * t + c) as u32).collect();
        let picked = h.reshape((b * t, dim))?.index_select(&Tensor::new(flat.as_slice(), &self.device)?, 0)?;
        self.log_probs(&picked)
    }

    /// Special-token id ([CLS], [SEP], [MASK], [PAD]=0, [UNK]).
    pub fn special(&self, name: &str) -> u32 {
        self.index.get(name).copied().unwrap_or(0)
    }

    fn log_probs(&self, hidden: &Tensor) -> Result<Tensor> {
        let h = self.head_norm.forward(&self.head_dense.forward(hidden)?.gelu_erf()?)?;
        candle_nn::ops::log_softmax(&self.decoder.forward(&h)?, D::Minus1)
    }

    /// Log-probability of each candidate (one or more words) filling `position`:
    /// as many [MASK]s as the candidate has pieces, summed.
    pub fn slot_scores(&self, words: &[String], position: usize, candidates: &[String]) -> Result<Vec<f32>> {
        let mut cache: HashMap<usize, (Vec<Vec<f32>>, usize)> = HashMap::new();
        let mut out = Vec::with_capacity(candidates.len());
        for cand in candidates {
            let pieces: Vec<String> = cand.split(' ').flat_map(|w| self.word_pieces(w)).collect();
            let k = pieces.len();
            if !cache.contains_key(&k) {
                let mut seq: Vec<String> = words[..position].to_vec();
                seq.extend(std::iter::repeat_n(MASK.to_string(), k));
                seq.extend(words[position + 1..].iter().cloned());
                let (ids, first) = self.encode(&seq);
                let at = first.get(position).copied().unwrap_or(ids.len());
                if at + k >= ids.len() {
                    cache.insert(k, (vec![], at));
                } else {
                    let h = self.hidden(&[ids])?.squeeze(0)?.narrow(0, at, k)?;
                    cache.insert(k, (self.log_probs(&h)?.to_vec2::<f32>()?, at));
                }
            }
            let (lp, _) = &cache[&k];
            if lp.is_empty() {
                out.push(f32::NEG_INFINITY);
                continue;
            }
            let unk = self.index[UNK];
            out.push(pieces.iter().enumerate().map(|(i, p)| {
                let id = self.index.get(p).copied().unwrap_or(unk);
                lp[i][id as usize] - if id == unk { 5.0 } else { 0.0 }
            }).sum());
        }
        Ok(out)
    }

    /// The `k` most likely whole words (single pieces with a letter) for `position`.
    pub fn predict(&self, words: &[String], position: usize, k: usize, visible: bool) -> Result<Vec<(String, f32)>> {
        let mut seq = words.to_vec();
        if !visible {
            seq[position] = MASK.into();
        }
        let (ids, first) = self.encode(&seq);
        let Some(&at) = first.get(position).filter(|&&a| a < ids.len()) else { return Ok(vec![]) };
        let lp = self.log_probs(&self.hidden(&[ids])?.squeeze(0)?.narrow(0, at, 1)?)?.squeeze(0)?.to_vec1::<f32>()?;
        let mut order: Vec<usize> = (0..lp.len())
            .filter(|&i| { let w = &self.vocab[i]; !w.starts_with("##") && !w.starts_with('[') && w.chars().any(char::is_alphabetic) })
            .collect();
        order.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
        Ok(order.into_iter().take(k).map(|i| (self.vocab[i].clone(), lp[i])).collect())
    }

    /// One vector per word (its first piece) for each sentence.
    pub fn contextual_batch(&self, sentences: &[Vec<String>]) -> Result<Vec<Vec<Vec<f32>>>> {
        let mut out = Vec::with_capacity(sentences.len());
        for chunk in sentences.chunks(32) {
            let enc: Vec<(Vec<u32>, Vec<usize>)> = chunk.iter().map(|s| self.encode(s)).collect();
            let rows: Vec<Vec<u32>> = enc.iter().map(|(ids, _)| ids.clone()).collect();
            let h = self.hidden(&rows)?;
            for (i, (ids, first)) in enc.iter().enumerate() {
                let m = h.get(i)?.to_vec2::<f32>()?;
                out.push(first.iter().filter(|&&f| f < ids.len()).map(|&f| m[f].clone()).collect());
            }
        }
        Ok(out)
    }

    /// Whether the word has its own vocabulary entry (not split, not unknown).
    pub fn knows(&self, word: &str) -> bool {
        self.index.contains_key(&word.to_lowercase())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    /// Compressed copies (half precision, 8-bit) score candidates almost exactly
    /// like the original, and keep the same ranking.
    #[test]
    fn compressed_models_score_like_the_original() {
        let dir = Path::new("data/bert-mini");
        if !dir.join("model.safetensors").exists() { return }
        let full = Bert::load(dir, Device::Cpu).unwrap();
        let weights = candle_core::safetensors::load(dir.join("model.safetensors"), &Device::Cpu).unwrap();
        let tmp = std::env::temp_dir().join("scribe-bert-compressed");
        for kind in ["f16", "q8"] {
            let d = tmp.join(kind);
            std::fs::create_dir_all(&d).unwrap();
            for f in ["vocab.txt", "config.json"] { std::fs::copy(dir.join(f), d.join(f)).unwrap(); }
            let _ = std::fs::remove_file(d.join("model.safetensors"));
            if kind == "f16" {
                let half: HashMap<String, Tensor> = weights.iter().map(|(k, v)| (k.clone(), v.to_dtype(DType::F16).unwrap())).collect();
                candle_core::safetensors::save(&half, d.join("model.safetensors")).unwrap();
            } else {
                candle_core::safetensors::save(&quantize_q8(&weights).unwrap(), d.join("model.q8.safetensors")).unwrap();
            }
            let small = Bert::load(&d, Device::Cpu).unwrap();
            let s = words("she drank a cup of hot tea before bed .");
            let cands = words("tea coffee gravel water soup");
            let (a, b) = (full.slot_scores(&s, 6, &cands).unwrap(), small.slot_scores(&s, 6, &cands).unwrap());
            for (x, y) in a.iter().zip(&b) { assert!((x - y).abs() < 0.25, "{kind}: {x} vs {y}") }
            let order = |v: &[f32]| { let mut i: Vec<usize> = (0..v.len()).collect(); i.sort_by(|&p, &q| v[q].total_cmp(&v[p])); i };
            assert_eq!(order(&a)[..2], order(&b)[..2], "{kind}: same top choices");
        }
    }

    /// Runs only where the converted weights exist (data/bert-mini).
    #[test]
    fn fills_in_obvious_words() {
        let dir = Path::new("data/bert-mini");
        if !dir.join("model.safetensors").exists() { return }
        let m = Bert::load(dir, Device::Cpu).unwrap();
        let top: Vec<String> = m.predict(&words("paris is the capital of france ."), 5, 5, false).unwrap().into_iter().map(|x| x.0).collect();
        eprintln!("paris is the capital of [MASK]: {top:?}");
        assert!(top.contains(&"france".to_string()), "{top:?}");
        let s = m.slot_scores(&words("she drank a cup of hot tea ."), 6, &words("tea gravel")).unwrap();
        assert!(s[0] > s[1], "{s:?}");
        assert_eq!(m.word_pieces("unaffable").concat().replace("##", ""), "unaffable");
        let v = m.contextual_batch(&[words("the bank of the river"), words("a bank loan")]).unwrap();
        assert_eq!((v[0].len(), v[1].len(), v[0][0].len()), (5, 3, 256));
        // A fresh model starts as BERT does: small weights, unit LayerNorm.
        let (_, fresh) = Bert::new_random(dir, Device::Cpu).unwrap();
        for (name, v) in fresh.data().lock().unwrap().iter() {
            let t = v.as_tensor().flatten_all().unwrap();
            let std = t.sqr().unwrap().mean_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
            if (name.contains("LayerNorm") || name.contains("final_norm")) && name.ends_with("weight") { assert!((std - 1.0).abs() < 1e-5, "{name}") }
            else if name.ends_with("bias") { assert!(std < 1e-6, "{name}") }
            else { assert!((0.015..0.025).contains(&std), "{name}: {std}") }
        }
        // The pre-LayerNorm layout builds, has its final norm, and runs.
        let tmp = std::env::temp_dir().join("scribe-bert-preln-test");
        std::fs::create_dir_all(&tmp).unwrap();
        let mut cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
        cfg["pre_layer_norm"] = serde_json::Value::Bool(true);
        std::fs::write(tmp.join("config.json"), cfg.to_string()).unwrap();
        std::fs::copy(dir.join("vocab.txt"), tmp.join("vocab.txt")).unwrap();
        let (pre, pv) = Bert::new_random(&tmp, Device::Cpu).unwrap();
        assert!(pv.data().lock().unwrap().keys().any(|k| k.contains("final_norm")));
        assert_eq!(pre.contextual_batch(&[words("a cup of tea")]).unwrap()[0].len(), 4);
        // The trainable copy computes the same thing.
        let (t, vars) = Bert::load_trainable(dir, Device::Cpu).unwrap();
        assert!(vars.all_vars().len() >= 70);
        let a = m.slot_scores(&words("she drank a cup of hot tea ."), 6, &words("tea")).unwrap()[0];
        let b = t.slot_scores(&words("she drank a cup of hot tea ."), 6, &words("tea")).unwrap()[0];
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        let rows = vec![t.encode_sentence(&words("the cat sat ."))];
        assert_eq!(t.masked_log_probs(&rows, &[(0, 2)]).unwrap().dims(), &[1, t.vocab_len()]);
    }
}
