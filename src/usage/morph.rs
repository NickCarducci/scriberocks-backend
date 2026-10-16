//! Base form + ending tokens ("running" → "run ##ing"). Small-data models
//! learn more from shared pieces (BabyLM 2025: morpheme segmentation gave
//! large gains), and the dictionary already knows every word's base form.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pieces for one word: [base] or [base, ##ending]. Irregular forms
/// (ran, children) get "##irr".
pub fn split_word(word: &str, lemmas: impl Fn(&str) -> Vec<String>) -> Vec<String> {
    let w = word.to_lowercase();
    let candidates = lemmas(&w);
    if candidates.iter().any(|l| *l == w) {
        return vec![w];
    }
    // Bases under 3 letters are function-word artifacts ("as" → "a"): keep whole.
    let Some(base) = candidates.into_iter().find(|l| !l.contains(' ') && l.len() >= 3) else { return vec![w] };
    let ending = if w.ends_with("ing") {
        "##ing"
    } else if w.ends_with("est") && !base.ends_with("est") {
        "##est"
    } else if w.ends_with("er") && !base.ends_with("er") && w.len() > base.len() {
        "##er"
    } else if w.ends_with('s') && w.starts_with(&base[..base.len().min(w.len()).saturating_sub(1).max(1)]) {
        "##s"
    } else if w.ends_with("ed") || (w.ends_with('d') && base.ends_with('e')) {
        "##ed"
    } else {
        "##irr"
    };
    vec![base, ending.to_string()]
}

/// Precomputed splits for every inflected word seen in training, saved
/// with the model so inference needs no dictionary.
#[derive(Serialize, Deserialize, Default, Clone)]
pub struct Morphology {
    splits: HashMap<String, Vec<String>>,
}

impl Morphology {
    pub fn build(words: impl Iterator<Item = String>, lemmas: impl Fn(&str) -> Vec<String>) -> Morphology {
        let mut splits = HashMap::new();
        for w in words {
            let w = w.to_lowercase();
            if splits.contains_key(&w) {
                continue;
            }
            let pieces = split_word(&w, &lemmas);
            if pieces.len() > 1 {
                splits.insert(w, pieces);
            }
        }
        Morphology { splits }
    }

    pub fn split(&self, word: &str) -> Vec<String> {
        if word.starts_with('[') && word.ends_with(']') {
            return vec![word.to_string()]; // [MASK] and other specials pass through
        }
        let w = word.to_lowercase();
        self.splits.get(&w).cloned().unwrap_or_else(|| vec![w])
    }

    /// Pieces for a whole sentence, plus each word's base-piece index.
    pub fn split_sentence(&self, words: &[String]) -> (Vec<String>, Vec<usize>) {
        let mut pieces = Vec::new();
        let mut base = Vec::with_capacity(words.len());
        for w in words {
            base.push(pieces.len());
            pieces.extend(self.split(w));
        }
        (pieces, base)
    }

    pub fn len(&self) -> usize {
        self.splits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.splits.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lemmas(w: &str) -> Vec<String> {
        match w {
            "running" => vec!["run".into()],
            "ran" => vec!["run".into()],
            "exhausted" => vec!["exhaust".into()],
            "cats" => vec!["cat".into()],
            "happiest" => vec!["happy".into()],
            "bigger" => vec!["big".into()],
            "goes" => vec!["go".into()],
            "children" => vec!["child".into()],
            "news" => vec!["news".into()],
            other => vec![other.into()],
        }
    }

    #[test]
    fn splits_regular_and_irregular_inflections() {
        let cases = [
            ("running", vec!["run", "##ing"]),
            ("exhausted", vec!["exhaust", "##ed"]),
            ("cats", vec!["cat", "##s"]),
            ("happiest", vec!["happy", "##est"]),
            ("bigger", vec!["big", "##er"]),
            ("ran", vec!["run", "##irr"]),
            ("children", vec!["child", "##irr"]),
        ];
        for (word, want) in cases {
            assert_eq!(split_word(word, lemmas), want, "{word}");
        }
    }

    #[test]
    fn short_function_words_stay_whole() {
        let eager = |w: &str| -> Vec<String> {
            match w { "as" => vec!["a".into()], "us" => vec!["u".into()], "was" => vec!["be".into()], "does" => vec!["do".into()], "goes" => vec!["go".into()], o => vec![o.into()] }
        };
        for w in ["as", "us", "was", "does", "goes"] {
            assert_eq!(split_word(w, eager), vec![w], "{w}");
        }
    }

    #[test]
    fn base_forms_and_unknown_words_stay_whole() {
        assert_eq!(split_word("news", lemmas), vec!["news"]);
        assert_eq!(split_word("zxqv", lemmas), vec!["zxqv"]);
        assert_eq!(split_word("Running", lemmas), vec!["run", "##ing"]);
    }

    #[test]
    fn table_splits_known_words_and_passes_others_through() {
        let m = Morphology::build(["running", "cats", "news", "cats"].iter().map(|s| s.to_string()), lemmas);
        assert_eq!(m.split("running"), vec!["run", "##ing"]);
        assert_eq!(m.split("news"), vec!["news"]);
        assert_eq!(m.split("unseen"), vec!["unseen"]);
        assert_eq!(m.len(), 2, "only words that actually split are stored");
    }

    #[test]
    fn splits_a_sentence_and_maps_words_to_their_base_piece() {
        let m = Morphology::build(["running", "cats"].iter().map(|s| s.to_string()), lemmas);
        let words: Vec<String> = ["the", "cats", "were", "running"].iter().map(|s| s.to_string()).collect();
        let (pieces, base) = m.split_sentence(&words);
        assert_eq!(pieces, vec!["the", "cat", "##s", "were", "run", "##ing"]);
        assert_eq!(base, vec![0, 1, 3, 4]);
    }
}
