//! Usage sentences for training: dictionary examples only — never definitions.

use crate::lexicon::Lexicon;

/// English usage examples from one line of the kaikki.org Wiktionary dump,
/// keeping sentences of `min_words..=max_words` words. Examples that carry an
/// "english" translation are foreign-language and skipped.
pub fn wiktionary_examples(line: &str, min_words: usize, max_words: usize) -> Vec<String> {
    let Ok(entry) = serde_json::from_str::<serde_json::Value>(line) else { return vec![] };
    if entry["lang_code"] != "en" {
        return vec![];
    }
    let mut out = Vec::new();
    for sense in entry["senses"].as_array().into_iter().flatten() {
        for ex in sense["examples"].as_array().into_iter().flatten() {
            if ex.get("english").is_some() {
                continue;
            }
            let Some(text) = ex["text"].as_str() else { continue };
            let text = text.trim();
            let n = text.split_whitespace().count();
            if (min_words..=max_words).contains(&n) && !out.iter().any(|t: &String| t == text) {
                out.push(text.to_string());
            }
        }
    }
    out
}

/// One WordNet example: the sentence, which sense it illustrates, and the
/// member word to look at.
pub struct SenseExample {
    pub synset: u32,
    pub word: String,
    pub sentence: String,
}

/// Every WordNet example sentence, labeled with its sense.
pub fn wordnet_examples(lexicon: &Lexicon) -> Vec<SenseExample> {
    let mut out = Vec::new();
    for (i, s) in lexicon.synsets.iter().enumerate() {
        for sentence in &s.examples {
            let lower = sentence.to_lowercase();
            // The member that actually appears (any inflection is resolved later).
            let word = s.members.iter().find(|m| lower.contains(&m.to_lowercase())).or(s.members.first());
            if let Some(word) = word {
                out.push(SenseExample { synset: i as u32, word: word.to_lowercase(), sentence: sentence.clone() });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINE: &str = r#"{"word":"run","lang":"English","lang_code":"en","pos":"verb","senses":[
        {"glosses":["To move swiftly."],"examples":[{"text":"I run every morning before work.","type":"example"},
                                                   {"text":"Run!","type":"example"},
                                                   {"text":"Ich laufe.","english":"I run.","type":"example"}]},
        {"glosses":["To manage."],"examples":[{"text":"She runs a small bakery in town.","type":"quotation"}]}]}"#;

    #[test]
    fn takes_english_examples_and_skips_definitions_and_short_lines() {
        let got = wiktionary_examples(LINE, 4, 60);
        assert_eq!(got, vec!["I run every morning before work.", "She runs a small bakery in town."]);
    }

    #[test]
    fn ignores_other_languages_and_bad_json() {
        assert!(wiktionary_examples(&LINE.replace(r#""lang_code":"en""#, r#""lang_code":"de""#), 4, 60).is_empty());
        assert!(wiktionary_examples("not json", 4, 60).is_empty());
    }

    #[test]
    fn caps_length() {
        assert!(wiktionary_examples(LINE, 4, 5).is_empty());
    }
}

/// Usage sentences (lowercased tokens): WordNet example sentences, treebank
/// sentences, and Wiktionary examples. Definitions are never included.
pub fn usage_sentences(lexicon: &Lexicon, treebank: &[Vec<String>], wiktionary: &[String]) -> Vec<Vec<String>> {
    let tokens = |s: &str| -> Vec<String> { crate::text::tokenize(s).into_iter().map(|w| w.to_lowercase()).collect() };
    let mut out: Vec<Vec<String>> = treebank.iter().map(|s| s.iter().map(|w| w.to_lowercase()).collect()).collect();
    out.extend(lexicon.synsets.iter().flat_map(|s| s.examples.iter()).map(|e| tokens(e)));
    out.extend(wiktionary.iter().map(|e| tokens(e)));
    out
}

/// Index of the token that is `word` (or an inflection of it, via `lemmas`).
pub fn find_word(tokens: &[String], word: &str, lemmas: impl Fn(&str) -> Vec<String>) -> Option<usize> {
    let word = word.to_lowercase();
    tokens
        .iter()
        .position(|t| t.to_lowercase() == word)
        .or_else(|| tokens.iter().position(|t| lemmas(&t.to_lowercase()).contains(&word)))
}

/// Deterministic 1-in-`n` holdout by (synset, example) so evaluation
/// sentences are never seen in training.
pub fn is_held_out(synset: u32, sentence: &str, n: u64) -> bool {
    let mut h: u64 = 1469598103934665603 ^ synset as u64;
    for b in sentence.bytes() {
        h = (h ^ b as u64).wrapping_mul(1099511628211);
    }
    h % n == 0
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    #[test]
    fn finds_exact_then_inflected_word() {
        let lemmas = |w: &str| if w == "ran" { vec!["run".to_string()] } else { vec![w.to_string()] };
        assert_eq!(find_word(&toks("she ran home"), "run", lemmas), Some(1));
        assert_eq!(find_word(&toks("Run fast"), "run", lemmas), Some(0));
        assert_eq!(find_word(&toks("walk home"), "run", lemmas), None);
    }

    #[test]
    fn holdout_is_stable_and_about_one_in_n() {
        assert_eq!(is_held_out(5, "a b", 10), is_held_out(5, "a b", 10));
        let held = (0..5000).filter(|i| is_held_out(*i, "same sentence", 10)).count();
        assert!((350..=650).contains(&held), "{held}");
    }
}

#[cfg(test)]
mod fluency_corpus_tests {
    use super::*;
    use crate::lexicon::{Pos, Synset};

    #[test]
    fn uses_examples_and_never_definitions() {
        let lex = Lexicon {
            categories: vec!["adj.all".into()],
            synsets: vec![Synset {
                id: "1".into(),
                pos: Pos::Adj,
                category: 0,
                definition: "filled with fear or apprehension".into(),
                examples: vec!["afraid of snakes".into()],
                members: vec!["afraid".into()],
                hypernyms: vec![],
                attributes: vec![],
                similar: vec![],
            }],
            lemmas: Default::default(),
            forms: Default::default(),
        };
        let corpus = usage_sentences(&lex, &[vec!["she".into(), "left".into()]], &["I run every day.".to_string()]);
        let flat: Vec<String> = corpus.iter().map(|s| s.join(" ")).collect();
        assert!(flat.contains(&"afraid of snakes".to_string()));
        assert!(flat.contains(&"she left".to_string()));
        assert!(flat.contains(&"i run every day .".to_string()));
        assert!(!flat.iter().any(|s| s.contains("apprehension")));
    }
}
