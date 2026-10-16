//! Sense modeling: a learned head that scores a word's candidate senses from
//! its contextual vector. Three parts, summed per candidate:
//!   - a vector per synset seen in training (sense-tagged usage, e.g. SemCor),
//!   - a vector per supersense (WordNet's 45 lexicographer categories), shared
//!     by every synset in it — so senses never seen in training still score,
//!   - a bias by dictionary rank (WordNet lists senses most-frequent first).
//! Trained with a softmax over each word's own candidates only.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Dictionary ranks past this share one bias.
pub const MAX_RANK: usize = 8;

/// One candidate sense of a word.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub synset: u32,
    /// Supersense (lexicographer category index).
    pub category: u16,
    /// Position in the dictionary's sense list (0 = first-listed).
    pub rank: usize,
}

pub struct Example {
    pub vector: Vec<f32>,
    pub candidates: Vec<Candidate>,
    /// Index into `candidates`.
    pub gold: usize,
}

pub struct TrainOptions {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub seed: u64,
}

impl Default for TrainOptions {
    fn default() -> Self {
        TrainOptions { epochs: 8, batch: 256, lr: 3e-3, weight_decay: 0.01, seed: 1 }
    }
}

#[derive(Serialize, Deserialize)]
struct Meta {
    dim: usize,
    categories: usize,
    /// synset → row in the synset table (row 0 = every synset not seen in training).
    rows: HashMap<u32, u32>,
}

pub struct SenseHead {
    meta: Meta,
    varmap: VarMap,
    synsets: Tensor,
    supersenses: Tensor,
    ranks: Tensor,
    device: Device,
}

impl SenseHead {
    fn build(meta: Meta, device: Device) -> Result<SenseHead> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let zero = candle_nn::Init::Const(0.0);
        let synsets = vb.get_with_hints((meta.rows.len() + 1, meta.dim), "synsets", zero)?;
        let supersenses = vb.get_with_hints((meta.categories, meta.dim), "supersenses", zero)?;
        let ranks = vb.get_with_hints(MAX_RANK, "ranks", zero)?;
        Ok(SenseHead { meta, varmap, synsets, supersenses, ranks, device })
    }

    fn row(&self, synset: u32) -> u32 {
        self.meta.rows.get(&synset).copied().unwrap_or(0)
    }

    /// Logits [batch, k] for padded candidate lists; padding gets -1e9.
    fn logits(&self, examples: &[&Example]) -> Result<Tensor> {
        let (b, d) = (examples.len(), self.meta.dim);
        let k = examples.iter().map(|e| e.candidates.len()).max().unwrap_or(1).max(1);
        let (mut rows, mut cats, mut ranks, mut pad, mut vecs) = (vec![], vec![], vec![], vec![], Vec::with_capacity(b * d));
        for e in examples {
            vecs.extend_from_slice(&e.vector);
            for j in 0..k {
                match e.candidates.get(j) {
                    Some(c) => {
                        rows.push(self.row(c.synset));
                        cats.push((c.category as u32).min(self.meta.categories as u32 - 1));
                        ranks.push(c.rank.min(MAX_RANK - 1) as u32);
                        pad.push(0.0f32);
                    }
                    None => {
                        rows.push(0);
                        cats.push(0);
                        ranks.push(0);
                        pad.push(-1e9);
                    }
                }
            }
        }
        let dev = &self.device;
        let h = Tensor::from_vec(vecs, (b, 1, d), dev)?;
        let rows = Tensor::from_vec(rows, b * k, dev)?;
        let cats = Tensor::from_vec(cats, b * k, dev)?;
        let senses = (self.synsets.index_select(&rows, 0)? + self.supersenses.index_select(&cats, 0)?)?.reshape((b, k, d))?;
        let dot = senses.broadcast_mul(&h)?.sum(D::Minus1)?;
        let bias = self.ranks.index_select(&Tensor::from_vec(ranks, b * k, dev)?, 0)?.reshape((b, k))?;
        dot + bias + Tensor::from_vec(pad, (b, k), dev)?
    }

    pub fn train(examples: &[Example], categories: usize, opts: &TrainOptions, device: Device) -> Result<SenseHead> {
        let dim = examples.first().map_or(1, |e| e.vector.len());
        let mut rows = HashMap::new();
        for e in examples {
            let s = e.candidates[e.gold].synset;
            let next = rows.len() as u32 + 1;
            rows.entry(s).or_insert(next);
        }
        let head = SenseHead::build(Meta { dim, categories, rows }, device)?;
        let mut opt = AdamW::new(head.varmap.all_vars(), ParamsAdamW { lr: opts.lr, weight_decay: opts.weight_decay, ..Default::default() })?;
        let mut rng = crate::usage::masking::Rng::new(opts.seed);
        let mut order: Vec<usize> = (0..examples.len()).filter(|&i| examples[i].candidates.len() > 1).collect();
        for _ in 0..opts.epochs {
            for i in (1..order.len()).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            for chunk in order.chunks(opts.batch) {
                let batch: Vec<&Example> = chunk.iter().map(|&i| &examples[i]).collect();
                let gold: Vec<u32> = batch.iter().map(|e| e.gold as u32).collect();
                let loss = candle_nn::loss::cross_entropy(&head.logits(&batch)?, &Tensor::new(gold.as_slice(), &head.device)?)?;
                opt.backward_step(&loss)?;
            }
        }
        Ok(head)
    }

    /// Score per candidate (higher is better).
    pub fn scores(&self, vector: &[f32], candidates: &[Candidate]) -> Result<Vec<f32>> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        let e = Example { vector: vector.to_vec(), candidates: candidates.to_vec(), gold: 0 };
        self.logits(&[&e])?.squeeze(0)?.to_vec1()
    }

    /// The best-scoring candidate's index.
    pub fn choose(&self, vector: &[f32], candidates: &[Candidate]) -> Result<Option<usize>> {
        let s = self.scores(vector, candidates)?;
        Ok((0..s.len()).max_by(|&a, &b| s[a].total_cmp(&s[b]).then(b.cmp(&a))))
    }

    pub fn knows(&self, synset: u32) -> bool {
        self.meta.rows.contains_key(&synset)
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        self.varmap.save(dir.join("senses.safetensors"))?;
        std::fs::write(dir.join("senses.json"), serde_json::to_vec(&self.meta).map_err(candle_core::Error::wrap)?)?;
        Ok(())
    }

    pub fn load(dir: &Path, device: Device) -> Result<SenseHead> {
        let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("senses.json"))?).map_err(candle_core::Error::wrap)?;
        let mut head = SenseHead::build(meta, device)?;
        head.varmap.load(dir.join("senses.safetensors"))?;
        Ok(head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(synset: u32, category: u16, rank: usize) -> Candidate {
        Candidate { synset, category, rank }
    }

    /// Noisy vector near `axis` in 4 dimensions.
    fn near(axis: usize, i: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..4).map(|d| ((i * 7 + d * 3) % 5) as f32 * 0.05).collect();
        v[axis] += 1.0;
        v
    }

    fn opts() -> TrainOptions {
        TrainOptions { epochs: 60, batch: 16, lr: 0.05, weight_decay: 0.0, seed: 3 }
    }

    #[test]
    fn learns_which_sense_a_context_points_to() {
        // "bank": river (10, cat 0) near axis 0, money (11, cat 1) near axis 1.
        let cands = vec![c(10, 0, 0), c(11, 1, 1)];
        let examples: Vec<Example> = (0..40).map(|i| Example { vector: near(i % 2, i), candidates: cands.clone(), gold: i % 2 }).collect();
        let head = SenseHead::train(&examples, 4, &opts(), Device::Cpu).unwrap();
        assert_eq!(head.choose(&near(0, 99), &cands).unwrap(), Some(0));
        assert_eq!(head.choose(&near(1, 99), &cands).unwrap(), Some(1));
    }

    #[test]
    fn unseen_senses_score_through_their_supersense() {
        // Train: synset 10 (cat 0) near axis 0, synset 11 (cat 1) near axis 1.
        let cands = vec![c(10, 0, 0), c(11, 1, 1)];
        let examples: Vec<Example> = (0..40).map(|i| Example { vector: near(i % 2, i), candidates: cands.clone(), gold: i % 2 }).collect();
        let head = SenseHead::train(&examples, 4, &opts(), Device::Cpu).unwrap();
        // A different word whose senses (20 in cat 0, 21 in cat 1) were never seen.
        let other = vec![c(21, 1, 0), c(20, 0, 1)];
        assert!(!head.knows(20) && !head.knows(21));
        assert_eq!(head.choose(&near(0, 5), &other).unwrap(), Some(1), "cat 0 context → synset 20");
        assert_eq!(head.choose(&near(1, 5), &other).unwrap(), Some(0), "cat 1 context → synset 21");
    }

    #[test]
    fn learns_that_first_listed_senses_are_most_common() {
        // Uninformative contexts; gold is the first-listed sense 80% of the time.
        let examples: Vec<Example> = (0..50)
            .map(|i| Example { vector: vec![0.0; 4], candidates: vec![c(100 + i as u32, 0, 0), c(200 + i as u32, 0, 1)], gold: (i % 5 == 0) as usize })
            .collect();
        let head = SenseHead::train(&examples, 1, &opts(), Device::Cpu).unwrap();
        assert_eq!(head.choose(&[0.0; 4], &[c(7, 0, 1), c(8, 0, 0)]).unwrap(), Some(1));
    }

    #[test]
    fn save_and_load_give_identical_scores() {
        let cands = vec![c(10, 0, 0), c(11, 1, 1)];
        let examples: Vec<Example> = (0..20).map(|i| Example { vector: near(i % 2, i), candidates: cands.clone(), gold: i % 2 }).collect();
        let head = SenseHead::train(&examples, 2, &opts(), Device::Cpu).unwrap();
        let dir = std::env::temp_dir().join(format!("sense-head-{}", std::process::id()));
        head.save(&dir).unwrap();
        let loaded = SenseHead::load(&dir, Device::Cpu).unwrap();
        assert_eq!(head.scores(&near(0, 1), &cands).unwrap(), loaded.scores(&near(0, 1), &cands).unwrap());
        let _ = std::fs::remove_dir_all(dir);
    }
}
