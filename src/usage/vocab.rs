//! Word-level vocabulary for the usage model.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const PAD: u32 = 0;
pub const UNK: u32 = 1;
pub const MASK: u32 = 2;
pub const CLS: u32 = 3;
pub const SPECIALS: [&str; 4] = ["[PAD]", "[UNK]", "[MASK]", "[CLS]"];

#[derive(Serialize, Deserialize, Clone)]
pub struct Vocab {
    words: Vec<String>,
    #[serde(skip)]
    index: HashMap<String, u32>,
}

impl Vocab {
    /// Words seen at least `min_count` times (lowercased), most frequent
    /// first, capped at `max_size` entries including the specials.
    pub fn build(sentences: &[Vec<String>], min_count: u32, max_size: usize) -> Vocab {
        let mut counts: HashMap<String, u32> = HashMap::new();
        for w in sentences.iter().flatten() {
            *counts.entry(w.to_lowercase()).or_default() += 1;
        }
        let mut ranked: Vec<(String, u32)> = counts.into_iter().filter(|(_, n)| *n >= min_count).collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let words: Vec<String> = SPECIALS
            .iter()
            .map(|s| s.to_string())
            .chain(ranked.into_iter().map(|(w, _)| w))
            .take(max_size)
            .collect();
        Vocab::from_words(words)
    }

    pub fn from_words(words: Vec<String>) -> Vocab {
        let index = words.iter().enumerate().map(|(i, w)| (w.clone(), i as u32)).collect();
        Vocab { words, index }
    }

    /// Rebuild the lookup table after deserializing.
    pub fn reindex(&mut self) {
        self.index = self.words.iter().enumerate().map(|(i, w)| (w.clone(), i as u32)).collect();
    }

    pub fn id(&self, word: &str) -> u32 {
        self.index.get(word).or_else(|| self.index.get(&word.to_lowercase())).copied().unwrap_or(UNK)
    }

    pub fn contains(&self, word: &str) -> bool {
        self.id(word) != UNK
    }

    pub fn encode<S: AsRef<str>>(&self, words: &[S]) -> Vec<u32> {
        words.iter().map(|w| self.id(w.as_ref())).collect()
    }

    pub fn word(&self, id: u32) -> &str {
        self.words.get(id as usize).map(String::as_str).unwrap_or("[UNK]")
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(words: &str) -> Vec<String> {
        words.split(' ').map(String::from).collect()
    }

    #[test]
    fn specials_come_first() {
        let v = Vocab::build(&[s("a b")], 1, 100);
        assert_eq!((v.id("[PAD]"), v.id("[UNK]"), v.id("[MASK]"), v.id("[CLS]")), (PAD, UNK, MASK, CLS));
    }

    #[test]
    fn keeps_frequent_words_lowercased_and_maps_rest_to_unk() {
        let v = Vocab::build(&[s("The cat sat"), s("the cat ran"), s("the dog")], 2, 100);
        assert_eq!(v.encode(&s("THE cat dog")), vec![v.id("the"), v.id("cat"), UNK]);
        assert!(v.contains("cat") && !v.contains("dog"));
    }

    #[test]
    fn caps_size_by_frequency() {
        let v = Vocab::build(&[s("a a a b b c")], 1, SPECIALS.len() + 2);
        assert!(v.contains("a") && v.contains("b") && !v.contains("c"));
        assert_eq!(v.len(), SPECIALS.len() + 2);
    }

    #[test]
    fn decodes_ids() {
        let v = Vocab::build(&[s("x y")], 1, 100);
        assert_eq!(v.word(v.id("y")), "y");
    }
}
