//! Word sense disambiguation from the dictionary alone (extended Lesk).
//!
//! Each sense gets a "signature": the content words of its definition,
//! examples, members, and its parent sense's definition. The sense whose
//! signature shares the most (rarity-weighted) words with the sentence wins,
//! with a small bonus for senses the dictionary lists first.

use crate::lexicon::{Lexicon, Pos};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct Disambiguator {
    /// Sorted word ids per synset.
    signatures: Vec<Vec<u32>>,
    idf: Vec<f32>,
    vocab: HashMap<String, u32>,
    /// Weight of "listed earlier" against definition overlap.
    pub prior: f32,
}

const STOP: &[&str] = &[
    "a", "an", "the", "of", "to", "in", "on", "for", "with", "by", "at", "from", "as", "or", "and", "but", "is", "are", "was", "were", "be",
    "been", "it", "its", "that", "this", "which", "who", "something", "someone", "somebody", "one", "into", "out", "up", "not", "no", "so",
    "have", "has", "had", "do", "does", "did", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "their", "our", "etc", "e.g.", "usually", "especially", "used", "make", "made", "being", "very", "more", "most",
];

pub fn content_words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() > 2 && !STOP.contains(&w.as_str()))
}

impl Disambiguator {
    /// `use_examples: false` holds example sentences out, for evaluating on them.
    pub fn new(lexicon: &Lexicon, use_examples: bool) -> Disambiguator {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut df: Vec<u32> = Vec::new();
        let id = |w: String, vocab: &mut HashMap<String, u32>, df: &mut Vec<u32>| -> u32 {
            let next = vocab.len() as u32;
            let i = *vocab.entry(w).or_insert(next);
            if i as usize == df.len() {
                df.push(0);
            }
            i
        };
        let base = |s: &crate::lexicon::Synset| -> Vec<String> {
            let mut words: Vec<String> = content_words(&s.definition).collect();
            if use_examples {
                for e in &s.examples {
                    words.extend(content_words(e));
                }
            }
            for m in &s.members {
                words.extend(content_words(m));
            }
            words
        };
        let mut signatures = Vec::with_capacity(lexicon.synsets.len());
        for s in &lexicon.synsets {
            let mut words = base(s);
            for &h in &s.hypernyms {
                words.extend(content_words(&lexicon.synsets[h as usize].definition));
                words.extend(lexicon.synsets[h as usize].members.iter().flat_map(|m| content_words(m).collect::<Vec<_>>()));
            }
            let mut sig = HashSet::new();
            for w in words {
                // Light stemming so "fear"/"fears"/"feared" meet.
                let stem = lexicon.lemmatize(&w, None).into_iter().next().unwrap_or(w);
                sig.insert(id(stem, &mut vocab, &mut df));
            }
            for &w in &sig {
                df[w as usize] += 1;
            }
            let mut sig: Vec<u32> = sig.into_iter().collect();
            sig.sort_unstable();
            signatures.push(sig);
        }
        let n = signatures.len() as f32;
        let idf = df.iter().map(|&d| (n / (1.0 + d as f32)).ln()).collect();
        Disambiguator { signatures, idf, vocab, prior: 1.0 }
    }

    pub fn load(path: &Path) -> Result<Disambiguator, Box<dyn std::error::Error>> {
        Ok(postcard::from_bytes(&std::fs::read(path)?)?)
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::write(path, postcard::to_allocvec(self)?)?;
        Ok(())
    }

    /// Picks a sense of `lemma` given the sentence's other content lemmas.
    /// Returns None when the dictionary has no sense for it.
    pub fn disambiguate(&self, lexicon: &Lexicon, lemma: &str, pos: Pos, context: &[String]) -> Option<u32> {
        self.disambiguate_with_support(lexicon, lemma, pos, context).map(|(s, _)| s)
    }

    /// Like `disambiguate`, plus whether the choice is supported: the word has
    /// a single sense, or the winning definition actually shares words with
    /// the sentence (not just the "listed first" bonus).
    pub fn disambiguate_with_support(&self, lexicon: &Lexicon, lemma: &str, pos: Pos, context: &[String]) -> Option<(u32, bool)> {
        let senses = lexicon.senses(lemma, pos);
        if senses.len() <= 1 {
            return senses.first().map(|&s| (s, true));
        }
        let ctx: HashSet<u32> = context.iter().filter(|w| w.as_str() != lemma).filter_map(|w| self.vocab.get(w).copied()).collect();
        senses
            .iter()
            .enumerate()
            .map(|(rank, &s)| {
                let overlap: f32 = self.signatures[s as usize]
                    .iter()
                    .filter(|w| ctx.contains(w))
                    .map(|&w| self.idf[w as usize])
                    .sum();
                (s, overlap, overlap + self.prior / (1.0 + rank as f32))
            })
            .max_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(s, overlap, _)| (s, overlap > 0.0))
    }
}

/// Held-out evaluation: disambiguate each dictionary example sentence's own
/// word, with examples excluded from all signatures. Returns
/// (cases, fine accuracy, first-sense fine accuracy, coarse accuracy,
/// first-sense coarse accuracy). Coarse = right dictionary category.
pub fn evaluate(lexicon: &Lexicon, prior: f32) -> (usize, f64, f64, f64, f64) {
    let mut wsd = Disambiguator::new(lexicon, false);
    wsd.prior = prior;
    let (mut n, mut hit, mut first, mut coarse, mut first_coarse) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for (sid, s) in lexicon.synsets.iter().enumerate() {
        for example in &s.examples {
            let words: Vec<String> = content_words(example).collect();
            let lemmas: Vec<String> = words.iter().map(|w| lexicon.lemmatize(w, None).into_iter().next().unwrap_or(w.clone())).collect();
            for member in s.members.iter().filter(|m| !m.contains(' ')) {
                let m = member.to_lowercase();
                let senses = lexicon.senses(&m, s.pos);
                if senses.len() < 2 || !lemmas.contains(&m) {
                    continue;
                }
                n += 1;
                first += (senses[0] as usize == sid) as usize;
                first_coarse += (lexicon.synsets[senses[0] as usize].category == s.category) as usize;
                let guess = wsd.disambiguate(lexicon, &m, s.pos, &lemmas);
                hit += (guess == Some(sid as u32)) as usize;
                coarse += guess.is_some_and(|g| lexicon.synsets[g as usize].category == s.category) as usize;
            }
        }
    }
    let pct = |x: usize| 100.0 * x as f64 / n as f64;
    (n, pct(hit), pct(first), pct(coarse), pct(first_coarse))
}
