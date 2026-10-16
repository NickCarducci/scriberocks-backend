//! Dictionary layer: Open English WordNet compiled into a compact binary.
//!
//! Every synset keeps its definition, examples, members, hypernyms and its
//! lexicographer category (`noun.feeling`, `verb.motion`, ...): 45 categories
//! the dictionary itself assigns, used downstream to sort rocks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pos {
    Noun,
    Verb,
    Adj,
    Adv,
}

impl Pos {
    fn from_wn(code: &str) -> Option<Pos> {
        match code {
            "n" => Some(Pos::Noun),
            "v" => Some(Pos::Verb),
            "a" | "s" => Some(Pos::Adj),
            "r" => Some(Pos::Adv),
            _ => None,
        }
    }

    /// Maps a Universal Dependencies tag to the dictionary's parts of speech.
    pub fn from_upos(upos: &str) -> Option<Pos> {
        match upos {
            "NOUN" | "PROPN" => Some(Pos::Noun),
            "VERB" | "AUX" => Some(Pos::Verb),
            "ADJ" => Some(Pos::Adj),
            "ADV" => Some(Pos::Adv),
            _ => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Synset {
    pub id: String,
    pub pos: Pos,
    /// Index into `Lexicon::categories`.
    pub category: u16,
    pub definition: String,
    pub examples: Vec<String>,
    pub members: Vec<String>,
    pub hypernyms: Vec<u32>,
    /// Adjective ↔ noun attribute links, e.g. afraid ↔ fear.
    pub attributes: Vec<u32>,
    /// Adjective clusters: tired ~ exhausted, weary, worn out.
    pub similar: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Lexicon {
    pub categories: Vec<String>,
    pub synsets: Vec<Synset>,
    /// lemma → senses per part of speech, most common sense first.
    pub lemmas: HashMap<String, Vec<(Pos, Vec<u32>)>>,
    /// irregular form → lemmas ("ran" → "run").
    pub forms: HashMap<String, Vec<String>>,
}

impl Lexicon {
    /// Builds from the extracted `english-wordnet-*-json` release directory.
    pub fn from_oewn_json(dir: &Path) -> Result<Lexicon, Box<dyn std::error::Error>> {
        use serde_json::Value;

        let mut categories = Vec::new();
        let mut synsets = Vec::new();
        let mut index: HashMap<String, u32> = HashMap::new();
        let mut raw: Vec<Value> = Vec::new();

        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        files.sort();

        // Pass 1: synsets from the lexicographer files.
        for path in &files {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            let is_lexfile = ["noun.", "verb.", "adj.", "adv."].iter().any(|p| name.starts_with(p));
            if !is_lexfile {
                continue;
            }
            let category = categories.len() as u16;
            categories.push(name);
            let map: serde_json::Map<String, Value> = serde_json::from_slice(&std::fs::read(path)?)?;
            for (id, value) in map {
                let pos = value["partOfSpeech"].as_str().and_then(Pos::from_wn).unwrap_or(Pos::Noun);
                index.insert(id.clone(), synsets.len() as u32);
                synsets.push(Synset {
                    id,
                    pos,
                    category,
                    definition: strings(&value["definition"]).into_iter().next().unwrap_or_default(),
                    examples: strings(&value["example"]),
                    members: strings(&value["members"]),
                    hypernyms: Vec::new(),
                    attributes: Vec::new(),
                    similar: Vec::new(),
                });
                raw.push(value);
            }
        }

        // Pass 2: resolve links now that every synset has an index.
        for (i, value) in raw.iter().enumerate() {
            let resolve = |key: &str| -> Vec<u32> {
                strings(&value[key]).iter().filter_map(|s| index.get(s).copied()).collect()
            };
            let mut hypernyms = resolve("hypernym");
            hypernyms.extend(resolve("instance_hypernym"));
            synsets[i].hypernyms = hypernyms;
            synsets[i].attributes = resolve("attribute");
            synsets[i].similar = resolve("similar");
        }

        // Pass 3: lemmas and irregular forms.
        let mut lemmas: HashMap<String, Vec<(Pos, Vec<u32>)>> = HashMap::new();
        let mut forms: HashMap<String, Vec<String>> = HashMap::new();
        for path in &files {
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            if !name.starts_with("entries-") {
                continue;
            }
            let map: serde_json::Map<String, Value> = serde_json::from_slice(&std::fs::read(path)?)?;
            for (lemma, by_pos) in map {
                let key = lemma.to_lowercase();
                let Some(by_pos) = by_pos.as_object() else { continue };
                for (pos_code, entry) in by_pos {
                    // Lemma keys can carry a homograph suffix like "-1"; take the POS letter.
                    let Some(pos) = Pos::from_wn(&pos_code[..1]) else { continue };
                    let senses: Vec<u32> = entry["sense"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|s| s["synset"].as_str()).filter_map(|s| index.get(s).copied()).collect())
                        .unwrap_or_default();
                    let slot = lemmas.entry(key.clone()).or_default();
                    match slot.iter_mut().find(|(p, _)| *p == pos) {
                        Some((_, existing)) => existing.extend(senses),
                        None => slot.push((pos, senses)),
                    }
                    for form in strings(&entry["form"]) {
                        let list = forms.entry(form.to_lowercase()).or_default();
                        if !list.contains(&key) {
                            list.push(key.clone());
                        }
                    }
                }
            }
        }

        Ok(Lexicon { categories, synsets, lemmas, forms })
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::write(path, postcard::to_allocvec(self)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Lexicon, Box<dyn std::error::Error>> {
        Ok(postcard::from_bytes(&std::fs::read(path)?)?)
    }

    /// Dictionary lemmas for a surface word, optionally restricted to a part
    /// of speech. Irregular forms first, then WordNet's suffix rules.
    pub fn lemmatize(&self, word: &str, pos: Option<Pos>) -> Vec<String> {
        let w = word.to_lowercase();
        let has = |lemma: &str| -> bool {
            self.lemmas.get(lemma).is_some_and(|ps| pos.is_none_or(|p| ps.iter().any(|(q, _)| *q == p)))
        };
        let mut out = Vec::new();
        let push = |s: String, out: &mut Vec<String>| {
            if !out.contains(&s) {
                out.push(s);
            }
        };
        if let Some(lemmas) = self.forms.get(&w) {
            for l in lemmas.iter().filter(|l| has(l)) {
                push(l.clone(), &mut out);
            }
        }
        if has(&w) {
            push(w.clone(), &mut out);
        }
        const NOUN: &[(&str, &str)] = &[("s", ""), ("ses", "s"), ("xes", "x"), ("zes", "z"), ("ches", "ch"), ("shes", "sh"), ("men", "man"), ("ies", "y")];
        const VERB: &[(&str, &str)] = &[("s", ""), ("ies", "y"), ("es", "e"), ("es", ""), ("ed", "e"), ("ed", ""), ("ing", "e"), ("ing", "")];
        const ADJ: &[(&str, &str)] = &[("er", ""), ("est", ""), ("er", "e"), ("est", "e"), ("ier", "y"), ("iest", "y")];
        let rules: Vec<&(&str, &str)> = match pos {
            Some(Pos::Noun) => NOUN.iter().collect(),
            Some(Pos::Verb) => VERB.iter().collect(),
            Some(Pos::Adj) => ADJ.iter().collect(),
            Some(Pos::Adv) => vec![],
            None => NOUN.iter().chain(VERB).chain(ADJ).collect(),
        };
        for (suffix, replacement) in rules {
            if let Some(stem) = w.strip_suffix(suffix) {
                if stem.is_empty() {
                    continue;
                }
                let candidate = format!("{stem}{replacement}");
                if has(&candidate) {
                    push(candidate, &mut out);
                } else if replacement.is_empty() {
                    // Doubled consonant: "stopped" → "stopp" → "stop".
                    // Character-wise: Wiktionary has words like "co₂s".
                    let mut rev = stem.chars().rev();
                    if let (Some(last), Some(prev)) = (rev.next(), rev.next()) {
                        let shorter = &stem[..stem.len() - last.len_utf8()];
                        if last == prev && last.is_ascii_alphabetic() && shorter.chars().count() >= 2 && has(shorter) {
                            push(shorter.to_string(), &mut out);
                        }
                    }
                }
            }
        }
        out
    }

    /// Senses of a lemma for a part of speech, most common first.
    pub fn senses(&self, lemma: &str, pos: Pos) -> &[u32] {
        self.lemmas
            .get(lemma)
            .and_then(|ps| ps.iter().find(|(p, _)| *p == pos))
            .map(|(_, s)| s.as_slice())
            .unwrap_or(&[])
    }

    /// Which parts of speech the dictionary allows for a surface word —
    /// used as a prior by the tagger.
    pub fn pos_set(&self, word: &str) -> Vec<Pos> {
        let mut out = Vec::new();
        for lemma in self.lemmatize(word, None) {
            if let Some(ps) = self.lemmas.get(&lemma) {
                for (p, _) in ps {
                    if !out.contains(p) {
                        out.push(*p);
                    }
                }
            }
        }
        out
    }

    pub fn category_name(&self, synset: u32) -> &str {
        &self.categories[self.synsets[synset as usize].category as usize]
    }
}

fn strings(value: &serde_json::Value) -> Vec<String> {
    value
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexicon(words: &[(&str, Pos)]) -> Lexicon {
        Lexicon {
            categories: vec![],
            synsets: vec![],
            lemmas: words.iter().map(|(w, p)| (w.to_string(), vec![(*p, vec![])])).collect(),
            forms: HashMap::new(),
        }
    }

    #[test]
    fn lemmatizes_regular_forms_and_doubled_consonants() {
        let lex = lexicon(&[("stop", Pos::Verb), ("cat", Pos::Noun)]);
        assert_eq!(lex.lemmatize("stopped", Some(Pos::Verb)), vec!["stop"]);
        assert_eq!(lex.lemmatize("cats", None), vec!["cat"]);
    }

    #[test]
    fn never_panics_on_non_ascii_words() {
        let lex = lexicon(&[("co", Pos::Noun)]);
        for w in ["co₂s", "₂₂ed", "naïveté", "café's", "日本ing", "ééed"] {
            let _ = lex.lemmatize(w, None);
            let _ = lex.lemmatize(w, Some(Pos::Verb));
        }
    }
}
