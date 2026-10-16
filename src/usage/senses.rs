//! Sense picking by example: each sense is the average contextual vector of
//! its word in that sense's example sentences; a new occurrence takes the
//! nearest sense (cosine).

use crate::lexicon::{Lexicon, Pos};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// How a sense's score was obtained, strongest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    /// The sense's own example sentences.
    Examples,
    /// Borrowed from WordNet relatives with examples (parent concept and its
    /// other children; for adjectives, the similar-adjective cluster).
    Relatives,
    /// Borrowed from the sense's dictionary category (noun.feeling, …).
    Category,
}

#[derive(Serialize, Deserialize, Default)]
pub struct SenseIndex {
    /// synset → (sum of centered example vectors, count)
    sums: HashMap<u32, (Vec<f32>, u32)>,
    /// Mean of all example vectors; subtracted from every vector so that the
    /// direction all vectors share doesn't swamp the differences between senses.
    #[serde(default)]
    center: Vec<f32>,
    /// LMMS-style backoff (Loureiro & Jorge 2019), rebuilt from the dictionary
    /// at load: parent synset → sum of unit centroids of its examples-bearing
    /// relatives; category → sum of unit centroids in that category.
    #[serde(skip)]
    relatives: HashMap<u32, (Vec<f32>, u32)>,
    #[serde(skip)]
    categories: HashMap<u16, (Vec<f32>, u32)>,
}

fn parents(lex: &Lexicon, synset: u32) -> Vec<u32> {
    let s = &lex.synsets[synset as usize];
    let mut out = s.hypernyms.clone();
    if s.pos == Pos::Adj {
        out.extend(&s.similar);
    }
    out
}

fn add_into(map_entry: &mut (Vec<f32>, u32), v: &[f32]) {
    if map_entry.0.is_empty() {
        map_entry.0 = vec![0.0; v.len()];
    }
    for (a, b) in map_entry.0.iter_mut().zip(v) {
        *a += b;
    }
    map_entry.1 += 1;
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n == 0.0 { v.to_vec() } else { v.iter().map(|x| x / n).collect() }
}

impl SenseIndex {
    pub fn with_center(center: Vec<f32>) -> SenseIndex {
        SenseIndex { center, ..Default::default() }
    }

    fn centered(&self, v: &[f32]) -> Vec<f32> {
        if self.center.len() == v.len() { v.iter().zip(&self.center).map(|(a, b)| a - b).collect() } else { v.to_vec() }
    }

    pub fn from_vectors(vectors: Vec<(u32, Vec<f32>)>) -> SenseIndex {
        let mut index = SenseIndex::default();
        for (synset, v) in vectors {
            index.add(synset, &v);
        }
        index
    }

    pub fn add(&mut self, synset: u32, v: &[f32]) {
        let v = &self.centered(v)[..];
        let entry = self.sums.entry(synset).or_insert_with(|| (vec![0.0; v.len()], 0));
        for (a, b) in entry.0.iter_mut().zip(v) {
            *a += b;
        }
        entry.1 += 1;
    }

    pub fn len(&self) -> usize {
        self.sums.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sums.is_empty()
    }

    pub fn knows(&self, synset: u32) -> bool {
        self.sums.contains_key(&synset)
    }

    /// Builds the relatives/category backoff from the dictionary graph.
    pub fn build_backoff(&mut self, lex: &Lexicon) {
        self.relatives.clear();
        self.categories.clear();
        let entries: Vec<(u32, Vec<f32>)> = self.sums.iter().map(|(&s, (sum, _))| (s, unit(sum))).collect();
        for (synset, u) in entries {
            if synset as usize >= lex.synsets.len() {
                continue;
            }
            add_into(self.relatives.entry(synset).or_default(), &u);
            for p in parents(lex, synset) {
                add_into(self.relatives.entry(p).or_default(), &u);
            }
            add_into(self.categories.entry(lex.synsets[synset as usize].category).or_default(), &u);
        }
    }

    /// Cosine between `v` and the sense, from its own examples if it has
    /// any, else its relatives, else its category. None without evidence
    /// (or before `build_backoff` for senses without examples).
    pub fn score(&self, synset: u32, v: &[f32], lex: &Lexicon) -> Option<(f32, Evidence)> {
        let q = self.centered(v);
        if let Some((sum, _)) = self.sums.get(&synset) {
            return Some((cosine(sum, &q), Evidence::Examples));
        }
        if synset as usize >= lex.synsets.len() {
            return None;
        }
        let mut borrowed = (Vec::new(), 0u32);
        for p in parents(lex, synset) {
            if let Some((sum, n)) = self.relatives.get(&p) {
                add_into(&mut borrowed, sum);
                borrowed.1 += n - 1;
            }
        }
        if borrowed.1 > 0 {
            return Some((cosine(&borrowed.0, &q), Evidence::Relatives));
        }
        let category = lex.synsets[synset as usize].category;
        self.categories.get(&category).map(|(sum, _)| (cosine(sum, &q), Evidence::Category))
    }

    /// Cosine between `v` and a sense's centroid, if the sense has examples.
    pub fn similarity(&self, synset: u32, v: &[f32]) -> Option<f32> {
        let v = self.centered(v);
        self.sums.get(&synset).map(|(sum, _)| cosine(sum, &v))
    }

    /// The candidate sense whose centroid is closest to `v`.
    pub fn best(&self, candidates: &[u32], v: &[f32]) -> Option<u32> {
        self.nearest(candidates, v, None)
    }

    /// As `best`, with one example (`held_out`, belonging to `synset`)
    /// removed from its centroid — for honest evaluation on the examples.
    pub fn best_excluding(&self, candidates: &[u32], v: &[f32], synset: u32, held_out: &[f32]) -> Option<u32> {
        self.nearest(candidates, v, Some((synset, held_out)))
    }

    fn nearest(&self, candidates: &[u32], v: &[f32], exclude: Option<(u32, &[f32])>) -> Option<u32> {
        let v = &self.centered(v)[..];
        let held: Option<Vec<f32>> = exclude.map(|(_, h)| self.centered(h));
        let exclude = exclude.map(|(s, _)| (s, held.as_deref().unwrap()));
        candidates
            .iter()
            .filter_map(|&c| {
                let (sum, n) = self.sums.get(&c)?;
                let (centroid, n): (Vec<f32>, u32) = match exclude {
                    Some((s, held)) if s == c => (sum.iter().zip(held).map(|(a, b)| a - b).collect(), n - 1),
                    _ => (sum.clone(), *n),
                };
                (n > 0).then(|| (c, cosine(&centroid, v)))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(c, _)| c)
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_centroid_wins() {
        let index = SenseIndex::from_vectors(vec![(1, vec![1.0, 0.0]), (1, vec![0.8, 0.2]), (2, vec![0.0, 1.0])]);
        assert_eq!(index.best(&[1, 2], &[0.9, 0.1]), Some(1));
        assert_eq!(index.best(&[1, 2], &[0.1, 0.9]), Some(2));
    }

    #[test]
    fn only_considers_given_candidates_and_known_senses() {
        let index = SenseIndex::from_vectors(vec![(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0])]);
        assert_eq!(index.best(&[2], &[1.0, 0.0]), Some(2));
        assert_eq!(index.best(&[3], &[1.0, 0.0]), None);
    }

    #[test]
    fn a_shared_center_is_subtracted_before_comparing() {
        // Both senses lean the same way; only the offset from the center tells them apart.
        let mut index = SenseIndex::with_center(vec![10.0, 10.0]);
        index.add(1, &[11.0, 10.0]);
        index.add(2, &[10.0, 11.0]);
        assert_eq!(index.best(&[1, 2], &[10.2, 10.9]), Some(2));
        assert!((index.similarity(1, &[11.0, 10.0]).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn similarity_is_cosine_to_the_centroid() {
        let index = SenseIndex::from_vectors(vec![(1, vec![2.0, 0.0])]);
        assert!((index.similarity(1, &[1.0, 0.0]).unwrap() - 1.0).abs() < 1e-6);
        assert_eq!(index.similarity(9, &[1.0, 0.0]), None);
    }

    #[test]
    fn leave_one_out_removes_an_example_from_its_centroid() {
        let index = SenseIndex::from_vectors(vec![(1, vec![1.0, 0.0]), (2, vec![0.0, 1.0]), (2, vec![0.9, 0.1])]);
        // With the [0.9, 0.1] example held out, sense 2 is just [0, 1], so 1 wins.
        assert_eq!(index.best_excluding(&[1, 2], &[0.9, 0.1], 2, &[0.9, 0.1]), Some(1));
    }
}

#[cfg(test)]
mod backoff_tests {
    use super::*;
    use crate::lexicon::{Lexicon, Pos, Synset};

    fn syn(id: u32, pos: Pos, category: u16, hypernyms: Vec<u32>, similar: Vec<u32>) -> Synset {
        Synset { id: id.to_string(), pos, category, definition: String::new(), examples: vec![], members: vec![], hypernyms, attributes: vec![], similar }
    }

    /// 0 emotion ← 1 fear (examples), 2 joy (examples), 3 dread (no examples)
    /// 4 machine ← 5 engine (examples); 6 gadget (no examples, no relatives with examples) in category 1
    /// 7 big (adj head, examples) ~ 8 huge (satellite, no examples)
    fn lexicon() -> Lexicon {
        Lexicon {
            categories: vec!["noun.feeling".into(), "noun.artifact".into(), "adj.all".into()],
            synsets: vec![
                syn(0, Pos::Noun, 0, vec![], vec![]),
                syn(1, Pos::Noun, 0, vec![0], vec![]),
                syn(2, Pos::Noun, 0, vec![0], vec![]),
                syn(3, Pos::Noun, 0, vec![0], vec![]),
                syn(4, Pos::Noun, 1, vec![], vec![]),
                syn(5, Pos::Noun, 1, vec![4], vec![]),
                syn(6, Pos::Noun, 1, vec![], vec![]),
                syn(7, Pos::Adj, 2, vec![], vec![8]),
                syn(8, Pos::Adj, 2, vec![], vec![7]),
            ],
            lemmas: Default::default(),
            forms: Default::default(),
        }
    }

    fn index() -> SenseIndex {
        let mut i = SenseIndex::from_vectors(vec![(1, vec![1.0, 0.0, 0.0]), (2, vec![0.8, 0.6, 0.0]), (5, vec![0.0, 0.0, 1.0]), (7, vec![0.0, 1.0, 0.0])]);
        i.build_backoff(&lexicon());
        i
    }

    #[test]
    fn senses_with_examples_use_them() {
        let (score, evidence) = index().score(1, &[1.0, 0.0, 0.0], &lexicon()).unwrap();
        assert!((score - 1.0).abs() < 1e-5);
        assert_eq!(evidence, Evidence::Examples);
    }

    #[test]
    fn a_sense_without_examples_borrows_from_its_relatives() {
        // dread (3) has no examples; its siblings under "emotion" (fear, joy) do.
        let (score, evidence) = index().score(3, &[0.9, 0.3, 0.0], &lexicon()).unwrap();
        assert_eq!(evidence, Evidence::Relatives);
        assert!(score > 0.9, "{score}");
        // Satellite adjective "huge" borrows from its head "big".
        assert_eq!(index().score(8, &[0.0, 1.0, 0.0], &lexicon()).unwrap().1, Evidence::Relatives);
    }

    #[test]
    fn otherwise_falls_back_to_the_dictionary_category() {
        // gadget (6) has no relatives with examples; noun.artifact has engine (5).
        let (score, evidence) = index().score(6, &[0.0, 0.0, 1.0], &lexicon()).unwrap();
        assert_eq!(evidence, Evidence::Category);
        assert!(score > 0.9);
    }

    #[test]
    fn backoff_is_rebuilt_after_loading_and_never_serialized() {
        let bytes = postcard::to_allocvec(&index()).unwrap();
        let mut loaded: SenseIndex = postcard::from_bytes(&bytes).unwrap();
        assert!(loaded.score(3, &[1.0, 0.0, 0.0], &lexicon()).is_none(), "no backoff until built");
        loaded.build_backoff(&lexicon());
        assert!(loaded.score(3, &[1.0, 0.0, 0.0], &lexicon()).is_some());
    }
}
