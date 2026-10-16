//! Classical sense head: IMS-style k-gram features (Zhong & Ng 2010) — the
//! surrounding words and the local collocations around a word — with one
//! averaged perceptron per lemma, trained on sense-tagged text (SemCor).
//! The baseline a learned head (usage vectors, FidelityAI) has to beat.

use crate::lexicon::Pos;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Collocation windows (from, to) relative to the word, as in IMS.
const COLLOCATIONS: &[(i32, i32)] = &[(-1, -1), (1, 1), (-2, -2), (2, 2), (-2, -1), (-1, 1), (1, 2), (-3, -1), (-2, 1), (-1, 2), (1, 3)];

/// Features of `words[i]` (lowercased words of its sentence).
pub fn features(words: &[String], i: usize) -> Vec<String> {
    let at = |k: i32| -> &str {
        let j = i as i32 + k;
        if j < 0 || j as usize >= words.len() { "_" } else { &words[j as usize] }
    };
    let mut out = vec!["bias".to_string()];
    for &(a, b) in COLLOCATIONS {
        let span: Vec<&str> = (a..=b).filter(|&k| k != 0).map(at).collect();
        out.push(format!("c{a},{b}={}", span.join(" ")));
    }
    let mut bag: Vec<&String> = words.iter().enumerate().filter(|&(j, w)| j != i && w.chars().any(char::is_alphabetic) && w.len() > 2).map(|(_, w)| w).collect();
    bag.sort();
    bag.dedup();
    out.extend(bag.into_iter().map(|w| format!("w={w}")));
    out
}

#[derive(Serialize, Deserialize, Default)]
struct LemmaModel {
    senses: Vec<u32>,
    /// feature → weight per sense (averaged after training).
    weights: HashMap<String, Vec<f32>>,
}

impl LemmaModel {
    fn scores(&self, feats: &[String]) -> Vec<f32> {
        let mut s = vec![0.0; self.senses.len()];
        for f in feats {
            if let Some(w) = self.weights.get(f) {
                for (a, b) in s.iter_mut().zip(w) {
                    *a += b;
                }
            }
        }
        s
    }
}

/// One training example: lemma, part of speech, gold sense, features.
pub struct Example {
    pub lemma: String,
    pub pos: Pos,
    pub sense: u32,
    pub features: Vec<String>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct KgramSenses {
    models: HashMap<(String, Pos), LemmaModel>,
}

impl KgramSenses {
    pub fn train(examples: &[Example], epochs: usize) -> KgramSenses {
        let mut groups: HashMap<(String, Pos), Vec<&Example>> = HashMap::new();
        for e in examples {
            groups.entry((e.lemma.clone(), e.pos)).or_default().push(e);
        }
        let models = groups.into_iter().map(|(key, group)| (key, train_lemma(&group, epochs))).collect();
        KgramSenses { models }
    }

    pub fn knows(&self, lemma: &str, pos: Pos) -> bool {
        self.models.contains_key(&(lemma.to_string(), pos))
    }

    /// Score per candidate sense (None for senses never seen in training, or
    /// for a lemma never seen at all).
    pub fn scores(&self, lemma: &str, pos: Pos, candidates: &[u32], feats: &[String]) -> Option<Vec<Option<f32>>> {
        let m = self.models.get(&(lemma.to_string(), pos))?;
        let s = m.scores(feats);
        Some(candidates.iter().map(|c| m.senses.iter().position(|x| x == c).map(|k| s[k])).collect())
    }

    /// The best-scoring candidate. The first-listed sense (most frequent in
    /// tagged text) stands unless the lemma was seen and another sense beats
    /// it by more than `margin`, or the first-listed sense was never seen.
    pub fn choose(&self, lemma: &str, pos: Pos, candidates: &[u32], feats: &[String], margin: f32) -> Option<u32> {
        let first = *candidates.first()?;
        let Some(scores) = self.scores(lemma, pos, candidates, feats) else { return Some(first) };
        let first_score = scores[0];
        let best = candidates.iter().zip(scores).filter_map(|(&c, s)| s.map(|s| (c, s))).max_by(|a, b| a.1.total_cmp(&b.1));
        Some(match (best, first_score) {
            (Some((c, s)), Some(f)) if s - f > margin => c,
            (Some((c, _)), None) => c,
            _ => first,
        })
    }
}

fn train_lemma(group: &[&Example], epochs: usize) -> LemmaModel {
    let mut senses: Vec<u32> = group.iter().map(|e| e.sense).collect();
    senses.sort();
    senses.dedup();
    let k = senses.len();
    let mut m = LemmaModel { senses, weights: HashMap::new() };
    if k < 2 {
        m.weights.insert("bias".into(), vec![1.0]);
        return m;
    }
    // Averaged perceptron with the lazy-sum trick: avg = w - u / c.
    let mut u: HashMap<String, Vec<f32>> = HashMap::new();
    let mut c = 1.0f32;
    // Deterministic shuffle per epoch.
    let mut order: Vec<usize> = (0..group.len()).collect();
    let mut seed = 0x9E3779B97F4A7C15u64;
    for _ in 0..epochs {
        for i in (1..order.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            order.swap(i, (seed % (i as u64 + 1)) as usize);
        }
        for &i in &order {
            let e = group[i];
            let gold = m.senses.iter().position(|&s| s == e.sense).unwrap();
            let s = m.scores(&e.features);
            let guess = (0..k).max_by(|&a, &b| s[a].total_cmp(&s[b]).then(b.cmp(&a))).unwrap();
            if guess != gold {
                for f in &e.features {
                    let w = m.weights.entry(f.clone()).or_insert_with(|| vec![0.0; k]);
                    let uu = u.entry(f.clone()).or_insert_with(|| vec![0.0; k]);
                    w[gold] += 1.0;
                    w[guess] -= 1.0;
                    uu[gold] += c;
                    uu[guess] -= c;
                }
            }
            c += 1.0;
        }
    }
    for (f, w) in m.weights.iter_mut() {
        if let Some(uu) = u.get(f) {
            for (a, b) in w.iter_mut().zip(uu) {
                *a -= b / c;
            }
        }
    }
    m.weights.retain(|_, w| w.iter().any(|x| *x != 0.0));
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    fn ex(sentence: &str, i: usize, sense: u32) -> Example {
        Example { lemma: "bank".into(), pos: Pos::Noun, sense, features: features(&words(sentence), i) }
    }

    #[test]
    fn features_include_collocations_and_surrounding_words() {
        let f = features(&words("she sat on the river bank today"), 5);
        assert!(f.contains(&"c-1,-1=river".to_string()));
        assert!(f.contains(&"c-2,-1=the river".to_string()));
        assert!(f.contains(&"c1,1=today".to_string()));
        assert!(f.contains(&"c1,3=today _ _".to_string()));
        assert!(f.contains(&"w=river".to_string()));
        assert!(!f.contains(&"w=bank".to_string()), "the word itself is not context");
        assert!(!f.contains(&"w=on".to_string()), "short function words are skipped");
    }

    #[test]
    fn learns_senses_from_context_words() {
        let train = vec![
            ex("fish swam near the river bank", 5, 1),
            ex("the muddy bank of the river flooded", 2, 1),
            ex("grass grew along the bank of the stream", 4, 1),
            ex("she deposited money at the bank downtown", 5, 2),
            ex("the bank approved the loan quickly", 1, 2),
            ex("money in the bank earns interest", 3, 2),
        ];
        let model = KgramSenses::train(&train, 10);
        let w = words("the river bank was muddy");
        assert_eq!(model.choose("bank", Pos::Noun, &[2, 1], &features(&w, 2), 0.0), Some(1));
        let w = words("he took money from the bank to repay a loan");
        assert_eq!(model.choose("bank", Pos::Noun, &[1, 2], &features(&w, 5), 0.0), Some(2));
        // A margin larger than any score difference keeps the first-listed sense.
        assert_eq!(model.choose("bank", Pos::Noun, &[1, 2], &features(&w, 5), 1e6), Some(1));
    }

    #[test]
    fn unknown_lemmas_and_unseen_senses_fall_back_to_first_listed() {
        let model = KgramSenses::train(&[ex("the river bank", 2, 1), ex("the money bank", 2, 1)], 5);
        let f = features(&words("the bank"), 1);
        assert_eq!(model.choose("shore", Pos::Noun, &[7, 8], &f, 0.0), Some(7));
        // Only sense 1 was seen: it wins over an unseen first-listed sense.
        assert_eq!(model.choose("bank", Pos::Noun, &[9, 1], &f, 0.0), Some(1));
        assert_eq!(model.scores("bank", Pos::Noun, &[9, 1], &f).unwrap()[0], None);
    }
}
