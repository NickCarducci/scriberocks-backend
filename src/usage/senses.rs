//! Sense picking by example: each sense is the average contextual vector of
//! its word in that sense's example sentences; a new occurrence takes the
//! nearest sense (cosine).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Default)]
pub struct SenseIndex {
    /// synset → (sum of centered example vectors, count)
    sums: HashMap<u32, (Vec<f32>, u32)>,
    /// Mean of all example vectors; subtracted from every vector so that the
    /// direction all vectors share doesn't swamp the differences between senses.
    #[serde(default)]
    center: Vec<f32>,
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
