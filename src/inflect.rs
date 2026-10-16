//! Inflection: lemma + Penn tag → surface form ("exhaust" + VBD → "exhausted").
//!
//! Irregular forms come from the EWT treebank (lemma, tag → form) and from
//! WordNet's irregular-form lists; everything else follows English spelling rules.

use crate::conllu::Sentence;
use crate::lexicon::Lexicon;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Serialize, Deserialize, Default)]
pub struct Inflector {
    /// "lemma TAG" → form, from treebank usage.
    irregular: HashMap<String, String>,
}

impl Inflector {
    pub fn train(sentences: &[Sentence], lexicon: &Lexicon) -> Inflector {
        let mut counts: HashMap<String, HashMap<String, u32>> = HashMap::new();
        for t in sentences.iter().flatten() {
            let lemma = t.lemma.to_lowercase();
            let form = t.form.to_lowercase();
            if matches!(t.xpos.as_str(), "VBD" | "VBN" | "VBZ" | "VBG" | "NNS" | "JJR" | "JJS" | "RBR" | "RBS") {
                *counts.entry(format!("{lemma} {}", t.xpos)).or_default().entry(form).or_default() += 1;
            }
        }
        // An irregular form wins only if it's the most common spelling for its slot
        // (so "gon" from "gonna" doesn't replace "going").
        let mut irregular: HashMap<String, String> = counts
            .into_iter()
            .filter_map(|(k, forms)| {
                let (lemma, tag) = k.split_once(' ')?;
                let regular_form = regular(lemma, tag);
                let (best, n) = forms.iter().max_by_key(|(f, n)| (**n, std::cmp::Reverse((*f).clone())))?;
                (*best != regular_form && *n >= 2).then(|| (k.clone(), best.clone()))
            })
            .collect();
        // WordNet lists irregular forms without saying which slot they fill.
        for (form, lemmas) in &lexicon.forms {
            for lemma in lemmas {
                for (pos, _) in lexicon.lemmas.get(lemma).into_iter().flatten() {
                    let slots: &[&str] = match pos {
                        crate::lexicon::Pos::Verb if form.ends_with("ing") => &["VBG"],
                        crate::lexicon::Pos::Verb if form.ends_with('s') && !form.ends_with("ss") => &["VBZ"],
                        crate::lexicon::Pos::Verb if form.ends_with('n') => &["VBN"],
                        crate::lexicon::Pos::Verb => &["VBD", "VBN"],
                        crate::lexicon::Pos::Noun => &["NNS"],
                        crate::lexicon::Pos::Adj if form.ends_with("st") => &["JJS"],
                        crate::lexicon::Pos::Adj => &["JJR"],
                        _ => &[],
                    };
                    for slot in slots {
                        irregular.entry(format!("{lemma} {slot}")).or_insert_with(|| form.clone());
                    }
                }
            }
        }
        Inflector { irregular }
    }

    pub fn load(path: &Path) -> Result<Inflector, Box<dyn std::error::Error>> {
        Ok(postcard::from_bytes(&std::fs::read(path)?)?)
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::write(path, postcard::to_allocvec(self)?)?;
        Ok(())
    }

    /// Inflects each word of a (possibly multiword) lemma's head: for verbs the
    /// first word ("worn out" stays, "give up" → "gave up"), for nouns the last.
    pub fn inflect(&self, lemma: &str, xpos: &str) -> String {
        let words: Vec<&str> = lemma.split(' ').collect();
        if words.len() > 1 {
            if !xpos.starts_with("VB") && !xpos.starts_with("NN") {
                return lemma.to_string(); // "more than", "worn out": degree forms don't apply
            }
            let head = if xpos.starts_with("VB") { 0 } else { words.len() - 1 };
            return words.iter().enumerate().map(|(i, w)| if i == head { self.inflect(w, xpos) } else { w.to_string() }).collect::<Vec<_>>().join(" ");
        }
        self.irregular.get(&format!("{lemma} {xpos}")).cloned().unwrap_or_else(|| regular(lemma, xpos))
    }
}

/// English spelling rules for regular inflection.
pub fn regular(lemma: &str, xpos: &str) -> String {
    let w = lemma;
    let ends = |s: &str| w.ends_with(s);
    let consonant_y = w.len() > 1 && ends("y") && !is_vowel(w.as_bytes()[w.len() - 2]);
    match xpos {
        "NNS" | "NNPS" | "VBZ" => {
            if ends("s") || ends("x") || ends("z") || ends("ch") || ends("sh") || (xpos == "VBZ" && ends("o") && w.len() > 1 && !is_vowel(w.as_bytes()[w.len() - 2])) {
                format!("{w}es")
            } else if consonant_y {
                format!("{}ies", &w[..w.len() - 1])
            } else {
                format!("{w}s")
            }
        }
        "VBG" => {
            if ends("ie") {
                format!("{}ying", &w[..w.len() - 2])
            } else if ends("ee") || ends("ye") || ends("oe") || w == "be" {
                format!("{w}ing")
            } else if ends("e") && w.len() > 2 {
                format!("{}ing", &w[..w.len() - 1])
            } else if doubles(w) {
                format!("{w}{}ing", last(w))
            } else {
                format!("{w}ing")
            }
        }
        "VBD" | "VBN" => {
            if ends("e") {
                format!("{w}d")
            } else if consonant_y {
                format!("{}ied", &w[..w.len() - 1])
            } else if doubles(w) {
                format!("{w}{}ed", last(w))
            } else {
                format!("{w}ed")
            }
        }
        "JJR" | "JJS" | "RBR" | "RBS" => {
            let (er, more) = if xpos.ends_with('R') { ("er", "more") } else { ("est", "most") };
            if syllables(w) > 2 || (syllables(w) == 2 && !consonant_y && !ends("le") && !ends("ow")) || xpos.starts_with("RB") && ends("ly") {
                format!("{more} {w}")
            } else if ends("e") {
                format!("{w}{}", &er[1..])
            } else if consonant_y {
                format!("{}i{er}", &w[..w.len() - 1])
            } else if doubles(w) {
                format!("{w}{}{er}", last(w))
            } else {
                format!("{w}{er}")
            }
        }
        _ => w.to_string(),
    }
}

fn is_vowel(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u')
}

fn last(w: &str) -> char {
    w.chars().last().unwrap_or(' ')
}

fn syllables(w: &str) -> usize {
    let b = w.as_bytes();
    let mut n = 0;
    let mut prev = false;
    for &c in b {
        let v = is_vowel(c) || c == b'y';
        if v && !prev {
            n += 1;
        }
        prev = v;
    }
    if w.ends_with('e') && !w.ends_with("le") && n > 1 {
        n -= 1;
    }
    n.max(1)
}

/// Single-syllable consonant–vowel–consonant words double the final consonant.
fn doubles(w: &str) -> bool {
    let b = w.as_bytes();
    b.len() >= 3
        && syllables(w) == 1
        && !is_vowel(b[b.len() - 1])
        && !matches!(b[b.len() - 1], b'w' | b'x' | b'y')
        && is_vowel(b[b.len() - 2])
        && !is_vowel(b[b.len() - 3])
}

#[cfg(test)]
mod tests {
    use super::regular;

    #[test]
    fn spelling_rules() {
        for (lemma, tag, form) in [
            ("run", "VBG", "running"),
            ("make", "VBG", "making"),
            ("see", "VBG", "seeing"),
            ("lie", "VBG", "lying"),
            ("visit", "VBD", "visited"),
            ("stop", "VBD", "stopped"),
            ("try", "VBD", "tried"),
            ("exhaust", "VBN", "exhausted"),
            ("box", "NNS", "boxes"),
            ("city", "NNS", "cities"),
            ("day", "NNS", "days"),
            ("go", "VBZ", "goes"),
            ("happy", "JJR", "happier"),
            ("big", "JJS", "biggest"),
            ("careful", "JJR", "more careful"),
            ("nice", "JJR", "nicer"),
        ] {
            assert_eq!(regular(lemma, tag), form, "{lemma} {tag}");
        }
    }
}
