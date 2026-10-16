//! Fluency scorer: an interpolated Kneser–Ney trigram model trained only on
//! dictionary definitions, dictionary examples and treebank sentences.
//! It never generates text; it judges how natural a candidate sentence reads.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

const D: f64 = 0.75;
const BOS: u32 = 0;
const EOS: u32 = 1;
const UNK: u32 = 2;

#[derive(Serialize, Deserialize, Default)]
pub struct LanguageModel {
    vocab: HashMap<String, u32>,
    tri: HashMap<(u32, u32, u32), u32>,
    /// c(u v •) and number of distinct w after (u v).
    ctx2: HashMap<(u32, u32), (u32, u32)>,
    /// N1+(• v w): distinct u before (v w).
    cont2: HashMap<(u32, u32), u32>,
    /// For v: Σ_w N1+(• v w) and number of distinct w with N1+(• v w) > 0.
    cont_ctx1: HashMap<u32, (u32, u32)>,
    /// N1+(• w): distinct v before w.
    cont1: HashMap<u32, u32>,
    cont1_total: u32,
    /// Distinct word types folded into <unk>; an unknown word's probability
    /// is shared among them, so rare words aren't scored like a common token.
    unk_types: u32,
}

impl LanguageModel {
    pub fn train(sentences: &[Vec<String>]) -> LanguageModel {
        let mut freq: HashMap<&str, u32> = HashMap::new();
        for s in sentences {
            for w in s {
                *freq.entry(w.as_str()).or_default() += 1;
            }
        }
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (w, id) in [("<s>", BOS), ("</s>", EOS), ("<unk>", UNK)] {
            vocab.insert(w.into(), id);
        }
        let mut words: Vec<&&str> = freq.iter().filter(|(_, n)| **n >= 2).map(|(w, _)| w).collect();
        words.sort();
        for w in words {
            let id = vocab.len() as u32;
            vocab.insert(w.to_string(), id);
        }

        let unk_types = freq.values().filter(|&&n| n < 2).count() as u32;
        let mut lm = LanguageModel { vocab, unk_types, ..Default::default() };
        for s in sentences {
            let ids = lm.ids(s);
            for win in ids.windows(3) {
                *lm.tri.entry((win[0], win[1], win[2])).or_default() += 1;
            }
        }
        let mut seen2 = std::collections::HashSet::new();
        for (&(u, v, w), &c) in &lm.tri {
            let e = lm.ctx2.entry((u, v)).or_default();
            e.0 += c;
            e.1 += 1;
            *lm.cont2.entry((v, w)).or_default() += 1;
            seen2.insert((v, w));
        }
        for &(v, w) in &seen2 {
            let n = lm.cont2[&(v, w)];
            let e = lm.cont_ctx1.entry(v).or_default();
            e.0 += n;
            e.1 += 1;
            *lm.cont1.entry(w).or_default() += 1;
        }
        lm.cont1_total = lm.cont1.values().sum();
        lm
    }

    /// Drops trigrams seen fewer than `min_count` times (and bigram
    /// continuations seen fewer than `min_count` times). Context totals are
    /// kept, so the removed mass goes to the lower-order estimate.
    pub fn prune(&mut self, min_count: u32) {
        self.tri.retain(|_, c| *c >= min_count);
        self.cont2.retain(|_, c| *c >= min_count);
        // Contexts with no surviving trigram fall back entirely to bigrams.
        let live: std::collections::HashSet<(u32, u32)> = self.tri.keys().map(|&(u, v, _)| (u, v)).collect();
        self.ctx2.retain(|k, _| live.contains(k));
    }

    fn ids(&self, words: &[String]) -> Vec<u32> {
        let mut ids = vec![BOS, BOS];
        ids.extend(words.iter().map(|w| *self.vocab.get(w).unwrap_or(&UNK)));
        ids.push(EOS);
        ids
    }

    fn p1(&self, w: u32) -> f64 {
        let n = self.cont1.get(&w).copied().unwrap_or(0) as f64;
        // Small uniform floor so unseen words keep some probability.
        0.99 * n / self.cont1_total.max(1) as f64 + 0.01 / self.vocab.len() as f64
    }

    fn p2(&self, v: u32, w: u32) -> f64 {
        match self.cont_ctx1.get(&v) {
            Some(&(total, types)) if total > 0 => {
                let c = self.cont2.get(&(v, w)).copied().unwrap_or(0) as f64;
                (c - D).max(0.0) / total as f64 + D * types as f64 / total as f64 * self.p1(w)
            }
            _ => self.p1(w),
        }
    }

    fn p3(&self, u: u32, v: u32, w: u32) -> f64 {
        match self.ctx2.get(&(u, v)) {
            Some(&(total, types)) if total > 0 => {
                let c = self.tri.get(&(u, v, w)).copied().unwrap_or(0) as f64;
                (c - D).max(0.0) / total as f64 + D * types as f64 / total as f64 * self.p2(v, w)
            }
            _ => self.p2(v, w),
        }
    }

    /// Mean log-probability per token (higher = more natural). Lowercased tokens.
    pub fn score(&self, words: &[String]) -> f64 {
        let ids = self.ids(words);
        let unk_share = (self.unk_types.max(1) as f64).ln();
        let lp: f64 = ids.windows(3).map(|w| self.p3(w[0], w[1], w[2]).ln() - if w[2] == UNK { unk_share } else { 0.0 }).sum();
        lp / (ids.len() - 2) as f64
    }

    pub fn load(path: &Path) -> Result<LanguageModel, Box<dyn std::error::Error>> {
        Ok(postcard::from_bytes(&std::fs::read(path)?)?)
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::write(path, postcard::to_allocvec(self)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<Vec<String>> {
        let lines = ["i am too tired", "i am too tired today", "she is too tired", "i am very happy", "a rare once only phrase"];
        lines.iter().cycle().take(40).chain(std::iter::once(&"the cat sat")).map(|l| l.split(' ').map(String::from).collect()).collect()
    }

    fn w(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    #[test]
    fn pruning_singletons_shrinks_the_model_but_keeps_its_preferences() {
        let full = LanguageModel::train(&corpus());
        let mut pruned = LanguageModel::train(&corpus());
        pruned.prune(2);
        assert!(pruned.tri.len() < full.tri.len());
        assert!(pruned.score(&w("i am too tired")) > pruned.score(&w("i am too sat")));
        assert!((pruned.score(&w("i am too tired")) - full.score(&w("i am too tired"))).abs() < 0.2);
        assert!(pruned.score(&w("the cat sat")).is_finite());
    }
}
