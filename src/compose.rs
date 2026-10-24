//! Compositional check for swaps (the pairing gate): does the meaning the
//! engine picked for a word fit the words it's attached to?
//!
//! Built from sense-tagged text that we parse (SemCor and the Princeton
//! WordNet Gloss Corpus): for every (meaning, grammatical role) the meanings
//! seen on the other end of that relation ("fix" in its repair meaning, as a
//! verb with an object: car, roof, problem…). A one-class check per (meaning,
//! role): a partner is typical when its hypernym ancestry overlaps enough
//! with some partner seen in training. A swap's word whose picked meaning is
//! atypical with its partners gets no swaps (its synonyms would come from
//! the wrong meaning: "fix run-on sentences" → "define run-on sentences").
use crate::engine::Sentence;
use crate::lexicon::Lexicon;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// How far up the hypernym tree a meaning's ancestry reaches.
const DEPTH: usize = 6;

#[derive(Default, Serialize, Deserialize)]
pub struct Compose {
    /// Role names: a base relation ("obj") when the meaning is the head,
    /// "obj^" when it is the dependent.
    pub roles: Vec<String>,
    /// (meaning, role, partner meanings seen), partners sorted and unique.
    pub entries: Vec<(u32, u16, Vec<u32>)>,
    #[serde(skip)]
    index: HashMap<(u32, u16), usize>,
}

/// Gate settings: an (meaning, role) needs `min_seen` partners before it
/// judges, and a partner is atypical below `typical` (absolute) or more than
/// `outcompeted` below another meaning of the same word (relative; 0 = off).
#[derive(Clone, Copy, Debug)]
pub struct Gate {
    pub min_seen: usize,
    pub typical: f32,
    pub outcompeted: f32,
}

impl Default for Gate {
    fn default() -> Self {
        Gate { min_seen: 3, typical: 0.0, outcompeted: 0.0 }
    }
}

/// The role a meaning plays in a relation: `deprel` is the dependent's
/// relation; `as_head` says which end the meaning is on.
pub fn role_name(deprel: &str, as_head: bool) -> String {
    let base = deprel.split(':').next().unwrap_or(deprel);
    if as_head { base.to_string() } else { format!("{base}^") }
}

/// A meaning and its hypernyms up to `DEPTH` levels, plus its category.
pub fn ancestry(lex: &Lexicon, s: u32) -> HashSet<u32> {
    let mut out = HashSet::from([s, u32::MAX - lex.synsets[s as usize].category as u32]);
    let mut frontier = vec![s];
    for _ in 0..DEPTH {
        frontier = frontier.iter().flat_map(|&x| lex.synsets[x as usize].hypernyms.iter().copied()).filter(|h| out.insert(*h)).collect();
        if frontier.is_empty() { break }
    }
    out
}

fn jaccard(a: &HashSet<u32>, b: &HashSet<u32>) -> f32 {
    let inter = a.intersection(b).count();
    inter as f32 / (a.len() + b.len() - inter).max(1) as f32
}

impl Compose {
    /// Accumulates pairs, then `finish` sorts them into entries.
    pub fn builder() -> ComposeBuilder {
        ComposeBuilder::default()
    }

    pub fn load(path: &Path) -> Result<Compose, Box<dyn std::error::Error>> {
        let mut c: Compose = postcard::from_bytes(&std::fs::read(path)?)?;
        c.reindex();
        Ok(c)
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::write(path, postcard::to_allocvec(self)?)?;
        Ok(())
    }

    fn reindex(&mut self) {
        self.index = self.entries.iter().enumerate().map(|(i, (s, r, _))| ((*s, *r), i)).collect();
    }

    pub fn partners(&self, meaning: u32, role: &str) -> &[u32] {
        let Some(r) = self.roles.iter().position(|x| x == role) else { return &[] };
        self.index.get(&(meaning, r as u16)).map_or(&[], |&i| self.entries[i].2.as_slice())
    }

    /// How typical `partner` is for `meaning` in `role` (best ancestry
    /// overlap with a partner seen in training), and how many were seen.
    /// None when this (meaning, role) was never seen.
    pub fn typicality(&self, lex: &Lexicon, meaning: u32, role: &str, partner: u32) -> Option<(f32, usize)> {
        let seen = self.partners(meaning, role);
        if seen.is_empty() { return None }
        let p = ancestry(lex, partner);
        let best = seen.iter().map(|&q| if q == partner { 1.0 } else { jaccard(&p, &ancestry(lex, q)) }).fold(0.0f32, f32::max);
        Some((best, seen.len()))
    }

    /// Whether the gate holds back swaps for token `i` of `s`, whose picked
    /// meaning is `meaning`: some partner with a meaning is atypical for it.
    pub fn blocks(&self, lex: &Lexicon, s: &Sentence, i: usize, meaning: u32, gate: Gate) -> bool {
        let t = &s.tokens[i];
        let mut partners: Vec<(String, u32)> = s.children(i).filter_map(|k| s.tokens[k].sense.map(|p| (role_name(&s.tokens[k].deprel, true), p))).collect();
        if let Some(h) = t.head {
            if let Some(p) = s.tokens[h].sense {
                partners.push((role_name(&t.deprel, false), p));
            }
        }
        let others: Vec<u32> = lex.lemmatize(&t.lemma, None).iter().flat_map(|l| lex.senses(l, lex.synsets[meaning as usize].pos).iter().copied()).filter(|&x| x != meaning).collect();
        partners.iter().any(|(role, p)| {
            let Some((typ, n)) = self.typicality(lex, meaning, role, *p) else { return false };
            if n < gate.min_seen { return false }
            if typ < gate.typical { return true }
            gate.outcompeted > 0.0
                && others.iter().filter_map(|&o| self.typicality(lex, o, role, *p).filter(|&(_, m)| m >= gate.min_seen)).any(|(alt, _)| alt - typ > gate.outcompeted)
        })
    }
}

#[derive(Default)]
pub struct ComposeBuilder {
    roles: Vec<String>,
    pairs: HashMap<(u32, u16), HashSet<u32>>,
}

impl ComposeBuilder {
    pub fn add(&mut self, meaning: u32, role: &str, partner: u32) {
        let r = match self.roles.iter().position(|x| x == role) {
            Some(r) => r,
            None => { self.roles.push(role.to_string()); self.roles.len() - 1 }
        };
        self.pairs.entry((meaning, r as u16)).or_default().insert(partner);
    }

    /// Both directions of a parsed arc between two tagged words.
    pub fn add_arc(&mut self, head: u32, dependent: u32, deprel: &str) {
        self.add(head, &role_name(deprel, true), dependent);
        self.add(dependent, &role_name(deprel, false), head);
    }

    pub fn finish(self) -> Compose {
        let mut entries: Vec<(u32, u16, Vec<u32>)> = self.pairs.into_iter().map(|((s, r), p)| {
            let mut p: Vec<u32> = p.into_iter().collect();
            p.sort_unstable();
            (s, r, p)
        }).collect();
        entries.sort_unstable_by_key(|e| (e.0, e.1));
        let mut c = Compose { roles: self.roles, entries, index: HashMap::new() };
        c.reindex();
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_name_the_end_of_the_relation() {
        assert_eq!(role_name("obj", true), "obj");
        assert_eq!(role_name("nmod:poss", false), "nmod^");
    }

    #[test]
    fn overlap_is_shared_over_combined() {
        let a: HashSet<u32> = [1, 2, 3].into();
        let b: HashSet<u32> = [2, 3, 4, 5].into();
        assert!((jaccard(&a, &b) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn builder_records_both_directions_and_survives_saving() {
        let mut b = Compose::builder();
        b.add_arc(10, 20, "obj");
        b.add_arc(10, 21, "obj");
        b.add_arc(10, 20, "obj");
        let c = b.finish();
        assert_eq!(c.partners(10, "obj"), &[20, 21]);
        assert_eq!(c.partners(20, "obj^"), &[10]);
        assert!(c.partners(10, "nsubj").is_empty());
        let path = std::env::temp_dir().join(format!("compose-test-{}.bin", std::process::id()));
        c.save(&path).unwrap();
        let back = Compose::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back.partners(10, "obj"), &[20, 21]);
    }
}
