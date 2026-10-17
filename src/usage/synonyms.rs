//! Synonym sets with shared usage samples: the "reflective" sense of a
//! synonym set is the set of real sentences where its members are
//! interchangeable. A thesaurus (WordNet synsets) proposes the members; usage
//! decides where they actually share a meaning — a sentence becomes a shared
//! sample when a scorer says other members fit its slot about as well as the
//! word that's there. Sentences using a word in another sense don't take the
//! other members, so they drop out on their own: no sense tagging needed.

use crate::engine::SlotScorer;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SharedSample {
    /// Index of the sentence in the corpus the sets were built from.
    pub sentence: usize,
    /// Position of the member in that sentence.
    pub slot: usize,
    /// Members that fit the slot (the original word first).
    pub fits: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SynonymSet {
    pub synset: u32,
    pub members: Vec<String>,
    pub samples: Vec<SharedSample>,
}

/// Members that fit `tokens[slot]`: the original word, and every other member
/// whose score is within `margin` (log-probability) of the original's.
pub fn fitting_members(scorer: &dyn SlotScorer, tokens: &[String], slot: usize, members: &[String], margin: f32) -> Vec<String> {
    let original = &tokens[slot];
    let Some(scores) = scorer.slot_scores(tokens, slot, members) else { return vec![] };
    let Some(base) = members.iter().position(|m| m == original).map(|i| scores[i]) else { return vec![] };
    let mut fits = vec![original.clone()];
    for (m, s) in members.iter().zip(&scores) {
        if m != original && s.is_finite() && *s >= base - margin {
            fits.push(m.clone());
        }
    }
    fits
}

/// Builds shared samples for each synset (members already lowercased,
/// single words). For each member, up to `per_member` corpus sentences
/// containing it are tried; a sentence is kept when at least one other
/// member fits.
pub fn build(synsets: &[(u32, Vec<String>)], corpus: &[Vec<String>], scorer: &dyn SlotScorer, per_member: usize, margin: f32) -> Vec<SynonymSet> {
    let wanted: std::collections::HashSet<&str> = synsets.iter().flat_map(|(_, m)| m.iter().map(String::as_str)).collect();
    let mut index: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
    for (si, sentence) in corpus.iter().enumerate() {
        for (pos, word) in sentence.iter().enumerate() {
            if wanted.contains(word.as_str()) {
                let list = index.entry(word.as_str()).or_default();
                if list.last().map(|(s, _)| *s) != Some(si) {
                    list.push((si, pos));
                }
            }
        }
    }
    synsets
        .iter()
        .map(|(synset, members)| {
            let mut samples = Vec::new();
            for member in members {
                for &(sentence, slot) in index.get(member.as_str()).into_iter().flatten().take(per_member) {
                    let fits = fitting_members(scorer, &corpus[sentence], slot, members, margin);
                    if fits.len() >= 2 {
                        samples.push(SharedSample { sentence, slot, fits });
                    }
                }
            }
            SynonymSet { synset: *synset, members: members.clone(), samples }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scores a candidate by whether it "belongs" next to the context words:
    /// money words around → financial members fit; river words → shore fits.
    struct FakeScorer;
    impl SlotScorer for FakeScorer {
        fn slot_scores(&self, context: &[String], _slot: usize, candidates: &[String]) -> Option<Vec<f32>> {
            let has = |w: &str| context.iter().any(|c| c == w);
            Some(
                candidates
                    .iter()
                    .map(|c| match c.as_str() {
                        "bank" => -1.0,
                        "depository" if has("money") => -1.5,
                        "shore" if has("river") => -1.2,
                        _ => -9.0,
                    })
                    .collect(),
            )
        }
    }

    fn words(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    #[test]
    fn other_members_fit_where_the_meaning_is_shared() {
        let fits = fitting_members(&FakeScorer, &words("she put money in the bank"), 5, &words("bank depository"), 2.0);
        assert_eq!(fits, words("bank depository"));
    }

    #[test]
    fn a_sentence_using_another_sense_is_not_a_shared_sample() {
        let corpus = vec![words("she put money in the bank"), words("we sat on the river bank"), words("the bank was closed")];
        let sets = build(&[(1, words("bank depository"))], &corpus, &FakeScorer, 10, 2.0);
        assert_eq!(sets[0].samples.len(), 1, "only the money sentence");
        assert_eq!(sets[0].samples[0], SharedSample { sentence: 0, slot: 5, fits: words("bank depository") });
    }

    #[test]
    fn the_margin_decides_how_close_a_fit_must_be() {
        let s = words("she put money in the bank");
        assert_eq!(fitting_members(&FakeScorer, &s, 5, &words("bank depository"), 0.1), words("bank"));
    }

    #[test]
    fn per_member_caps_the_sentences_tried() {
        let corpus: Vec<Vec<String>> = (0..20).map(|_| words("money in the bank")).collect();
        let sets = build(&[(1, words("bank depository"))], &corpus, &FakeScorer, 3, 2.0);
        assert_eq!(sets[0].samples.len(), 3);
    }
}
