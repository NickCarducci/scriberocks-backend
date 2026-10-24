//! Part-of-speech tagger: an averaged perceptron trained from scratch on the
//! UD English EWT treebank. The only "pretraining" is the dictionary: each
//! word's WordNet parts of speech are fed in as features.
//!
//! Trains on either column: UPOS (17 universal tags) or XPOS (Penn tags,
//! which carry tense and number: VBD, VBG, NNS…, needed for inflection).

use crate::conllu::Sentence;
use crate::lexicon::{Lexicon, Pos};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Column {
    Upos,
    Xpos,
}

impl Column {
    pub fn of(self, t: &crate::conllu::Token) -> &str {
        match self {
            Column::Upos => &t.upos,
            Column::Xpos => &t.xpos,
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
pub struct Tagger {
    tags: Vec<String>,
    weights: HashMap<String, Vec<f32>>,
    /// Frequent words that always take one tag in training.
    tagdict: HashMap<String, u16>,
}

/// Running totals for weight averaging.
#[derive(Default)]
struct Averager {
    totals: HashMap<String, Vec<f64>>,
    stamps: HashMap<String, Vec<u32>>,
}

impl Tagger {
    /// Drops weights smaller than `threshold` in magnitude, and features left
    /// with none (for shipping a smaller model). Returns (features, weights) kept.
    pub fn prune(&mut self, threshold: f32) -> (usize, usize) {
        self.weights.retain(|_, w| {
            for x in w.iter_mut() { if x.abs() < threshold { *x = 0.0 } }
            w.iter().any(|&x| x != 0.0)
        });
        (self.weights.len(), self.weights.values().map(|w| w.iter().filter(|&&x| x != 0.0).count()).sum())
    }

    /// Tags, the single-tag word dictionary, and every feature's weights
    /// (for exporting the tagger to other runtimes).
    pub fn parts(&self) -> (&[String], &HashMap<String, u16>, &HashMap<String, Vec<f32>>) {
        (&self.tags, &self.tagdict, &self.weights)
    }

    pub fn load(path: &Path) -> Result<Tagger, Box<dyn std::error::Error>> {
        Ok(postcard::from_bytes(&std::fs::read(path)?)?)
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::write(path, postcard::to_allocvec(self)?)?;
        Ok(())
    }

    pub fn tag<'a>(&'a self, words: &[&str], lexicon: Option<&Lexicon>) -> Vec<&'a str> {
        let context = context(words);
        let (mut p1, mut p2) = ("-START-", "-START2-");
        let mut out = Vec::with_capacity(words.len());
        for i in 0..words.len() {
            let tag = match self.tagdict.get(&normalize(words[i])) {
                Some(&t) => self.tags[t as usize].as_str(),
                None => {
                    let feats = features(i, words[i], &context, p1, p2, lexicon);
                    self.tags[self.predict(&feats)].as_str()
                }
            };
            out.push(tag);
            p2 = p1;
            p1 = tag;
        }
        out
    }

    fn predict(&self, feats: &[String]) -> usize {
        let n = self.tags.len();
        let mut scores = vec![0f32; n];
        for f in feats {
            if let Some(w) = self.weights.get(f) {
                for (s, v) in scores.iter_mut().zip(w) {
                    *s += v;
                }
            }
        }
        // Ties break toward the earlier tag, deterministic across runs.
        (0..n).fold(0, |best, t| if scores[t] > scores[best] { t } else { best })
    }

    pub fn train(sentences: &[Sentence], column: Column, lexicon: Option<&Lexicon>, iterations: usize) -> Tagger {
        let mut tags: Vec<String> = sentences.iter().flatten().map(|t| column.of(t).to_string()).collect();
        tags.sort();
        tags.dedup();
        let tag_index = |tag: &str| tags.iter().position(|t| t == tag).unwrap_or(0);
        let mut tagger = Tagger { tagdict: build_tagdict(sentences, column, &tag_index), tags: tags.clone(), ..Default::default() };
        let mut avg = Averager::default();
        let mut step: u32 = 0;
        let mut order: Vec<usize> = (0..sentences.len()).collect();
        let mut rng = 0x2545F4914F6CDD1Du64;

        for iter in 0..iterations {
            let (mut correct, mut total) = (0usize, 0usize);
            for &si in &order {
                let sentence = &sentences[si];
                let words: Vec<&str> = sentence.iter().map(|t| t.form.as_str()).collect();
                let context = context(&words);
                let (mut p1, mut p2) = ("-START-".to_string(), "-START2-".to_string());
                for (i, token) in sentence.iter().enumerate() {
                    let gold = tag_index(column.of(token));
                    let guess = match tagger.tagdict.get(&normalize(&token.form)) {
                        Some(&t) => t as usize,
                        None => {
                            step += 1;
                            let feats = features(i, &token.form, &context, &p1, &p2, lexicon);
                            let guess = tagger.predict(&feats);
                            if guess != gold {
                                for f in &feats {
                                    tagger.update(&mut avg, step, f, gold, 1.0);
                                    tagger.update(&mut avg, step, f, guess, -1.0);
                                }
                            }
                            guess
                        }
                    };
                    correct += (guess == gold) as usize;
                    total += 1;
                    p2 = std::mem::replace(&mut p1, tagger.tags[guess].clone());
                }
            }
            eprintln!("  iteration {}: train accuracy {:.2}%", iter + 1, 100.0 * correct as f64 / total as f64);
            // Fisher–Yates with xorshift so runs are reproducible.
            for i in (1..order.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                order.swap(i, (rng % (i as u64 + 1)) as usize);
            }
        }

        // Replace each weight with its average over all steps.
        let n = tagger.tags.len();
        for (feat, weights) in tagger.weights.iter_mut() {
            let (Some(totals), Some(stamps)) = (avg.totals.get(feat), avg.stamps.get(feat)) else { continue };
            for t in 0..n {
                let total = totals[t] + (step - stamps[t]) as f64 * weights[t] as f64;
                weights[t] = (total / step.max(1) as f64) as f32;
            }
        }
        tagger.weights.retain(|_, w| w.iter().any(|v| v.abs() > 1e-4));
        tagger
    }

    fn update(&mut self, avg: &mut Averager, step: u32, feat: &str, tag: usize, delta: f32) {
        let n = self.tags.len();
        let w = self.weights.entry(feat.to_string()).or_insert_with(|| vec![0.0; n]);
        let totals = avg.totals.entry(feat.to_string()).or_insert_with(|| vec![0.0; n]);
        let stamps = avg.stamps.entry(feat.to_string()).or_insert_with(|| vec![0; n]);
        totals[tag] += (step - stamps[tag]) as f64 * w[tag] as f64;
        stamps[tag] = step;
        w[tag] += delta;
    }

    pub fn accuracy(&self, sentences: &[Sentence], column: Column, lexicon: Option<&Lexicon>) -> f64 {
        let (mut correct, mut total) = (0usize, 0usize);
        for s in sentences {
            let words: Vec<&str> = s.iter().map(|t| t.form.as_str()).collect();
            for (guess, gold) in self.tag(&words, lexicon).iter().zip(s) {
                correct += (*guess == column.of(gold)) as usize;
                total += 1;
            }
        }
        100.0 * correct as f64 / total as f64
    }
}

fn normalize(word: &str) -> String {
    if word.contains('-') && word.chars().next().is_some_and(|c| c != '-') {
        "!HYPHEN".into()
    } else if word.chars().all(|c| c.is_ascii_digit()) && word.len() == 4 {
        "!YEAR".into()
    } else if word.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        "!DIGITS".into()
    } else {
        word.to_lowercase()
    }
}

fn context(words: &[&str]) -> Vec<String> {
    let mut c = vec!["-START-".to_string(), "-START2-".to_string()];
    c.extend(words.iter().map(|w| normalize(w)));
    c.push("-END-".into());
    c.push("-END2-".into());
    c
}

fn suffix(s: &str, n: usize) -> &str {
    let start = s.char_indices().rev().nth(n.saturating_sub(1)).map(|(i, _)| i).unwrap_or(0);
    &s[start..]
}

fn features(i: usize, word: &str, ctx: &[String], p1: &str, p2: &str, lexicon: Option<&Lexicon>) -> Vec<String> {
    let i = i + 2;
    let w = &ctx[i];
    let shape = if word.chars().next().is_some_and(char::is_uppercase) { "Cap" } else { "low" };
    let mut f = vec![
        "bias".to_string(),
        format!("w {w}"),
        format!("suf {}", suffix(w, 3)),
        format!("suf2 {}", suffix(w, 2)),
        format!("pre {}", w.chars().next().unwrap_or(' ')),
        format!("shape {shape}"),
        format!("p1 {p1}"),
        format!("p2 {p2}"),
        format!("p1p2 {p1} {p2}"),
        format!("p1w {p1} {w}"),
        format!("w-1 {}", ctx[i - 1]),
        format!("suf-1 {}", suffix(&ctx[i - 1], 3)),
        format!("w-2 {}", ctx[i - 2]),
        format!("w+1 {}", ctx[i + 1]),
        format!("suf+1 {}", suffix(&ctx[i + 1], 3)),
        format!("w+2 {}", ctx[i + 2]),
    ];
    if let Some(lex) = lexicon {
        // Dictionary prior: which parts of speech WordNet allows for this word.
        let mut set: Vec<&str> = lex
            .pos_set(word)
            .into_iter()
            .map(|p| match p {
                Pos::Noun => "n",
                Pos::Verb => "v",
                Pos::Adj => "a",
                Pos::Adv => "r",
            })
            .collect();
        set.sort();
        let set = if set.is_empty() { "none".to_string() } else { set.join("") };
        f.push(format!("dict {set}"));
        f.push(format!("dict-p1 {set} {p1}"));
    }
    f
}

fn build_tagdict(sentences: &[Sentence], column: Column, tag_index: &dyn Fn(&str) -> usize) -> HashMap<String, u16> {
    let mut counts: HashMap<String, HashMap<usize, u32>> = HashMap::new();
    for s in sentences {
        for t in s {
            *counts.entry(normalize(&t.form)).or_default().entry(tag_index(column.of(t))).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .filter_map(|(word, c)| {
            let total: u32 = c.values().sum();
            let (&best, &n) = c.iter().max_by_key(|(t, n)| (**n, std::cmp::Reverse(**t)))?;
            (total >= 20 && n as f64 / total as f64 >= 0.97).then_some((word, best as u16))
        })
        .collect()
}

#[cfg(test)]
mod prune_tests {
    use super::*;

    #[test]
    fn pruning_drops_small_weights_and_empty_features() {
        let mut t = Tagger { tags: vec!["A".into(), "B".into()], ..Default::default() };
        t.weights.insert("big".into(), vec![2.0, 0.1]);
        t.weights.insert("small".into(), vec![0.2, -0.3]);
        assert_eq!(t.prune(0.5), (1, 1));
        assert_eq!(t.weights["big"], vec![2.0, 0.0]);
    }
}
