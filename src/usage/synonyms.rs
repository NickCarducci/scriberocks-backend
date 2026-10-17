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

/// One use of a member in the corpus: where, and its Penn tag (so other
/// members can be put in the same form: "duties" → "obligations").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Occurrence<'a> {
    pub sentence: usize,
    pub slot: usize,
    pub xpos: &'a str,
}

/// Members that fit `tokens[slot]` (which holds `original` in some form):
/// the original, and every other member that — inflected like the original —
/// scores within `margin` (log-probability) of it. Returns member lemmas.
pub fn fitting_members(
    scorer: &dyn SlotScorer,
    tokens: &[String],
    slot: usize,
    original: &str,
    members: &[String],
    surface: &dyn Fn(&str) -> String,
    margin: f32,
) -> Vec<String> {
    // The original as written; the others in its form.
    let forms: Vec<String> = members.iter().map(|m| if m == original { tokens[slot].clone() } else { surface(m) }).collect();
    let Some(scores) = scorer.slot_scores(tokens, slot, &forms) else { return vec![] };
    let Some(base) = members.iter().position(|m| m == original).map(|i| scores[i]) else { return vec![] };
    let mut fits = vec![original.to_string()];
    for (m, s) in members.iter().zip(&scores) {
        if m != original && s.is_finite() && *s >= base - margin {
            fits.push(m.clone());
        }
    }
    fits
}

/// Builds shared samples for each synset. `occurrences(member)` lists the
/// member's uses (already restricted to the synset's part of speech by the
/// caller); `inflect(lemma, xpos)` puts another member in the same form. Up
/// to `per_member` uses per member are tried; a use is kept when at least one
/// other member fits.
pub fn build<'a>(
    synsets: &[(u32, Vec<String>)],
    corpus: &[Vec<String>],
    occurrences: &dyn Fn(u32, &str) -> Vec<Occurrence<'a>>,
    inflect: &dyn Fn(&str, &str) -> String,
    scorer: &dyn SlotScorer,
    per_member: usize,
    margin: f32,
) -> Vec<SynonymSet> {
    synsets
        .iter()
        .map(|(synset, members)| {
            let mut samples = Vec::new();
            for member in members {
                for occ in occurrences(*synset, member).into_iter().take(per_member) {
                    let surface = |lemma: &str| inflect(lemma, occ.xpos);
                    let fits = fitting_members(scorer, &corpus[occ.sentence], occ.slot, member, members, &surface, margin);
                    if fits.len() >= 2 {
                        samples.push(SharedSample { sentence: occ.sentence, slot: occ.slot, fits });
                    }
                }
            }
            SynonymSet { synset: *synset, members: members.clone(), samples }
        })
        .collect()
}

/// Thesaurus members worth offering: words of 2+ letters (hyphens and
/// apostrophes allowed), not on the caller's stoplist (e.g. the commonest
/// function words).
pub fn usable_member(member: &str, stop: &dyn Fn(&str) -> bool) -> bool {
    member.chars().filter(|c| c.is_alphabetic()).count() >= 2
        && member.chars().all(|c| c.is_alphabetic() || c == '-' || c == '\'')
        && !stop(member)
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

    /// Exact-token occurrences, base form (tag "NN"), for tests.
    fn exact<'a>(corpus: &'a [Vec<String>]) -> impl Fn(u32, &str) -> Vec<Occurrence<'a>> + 'a {
        move |_, member| {
            corpus
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.iter().position(|w| w == member).map(|slot| Occurrence { sentence: i, slot, xpos: "NN" }))
                .collect()
        }
    }

    fn base(lemma: &str, _xpos: &str) -> String {
        lemma.to_string()
    }

    #[test]
    fn other_members_fit_where_the_meaning_is_shared() {
        let s = words("she put money in the bank");
        let fits = fitting_members(&FakeScorer, &s, 5, "bank", &words("bank depository"), &|l| l.to_string(), 2.0);
        assert_eq!(fits, words("bank depository"));
    }

    #[test]
    fn a_sentence_using_another_sense_is_not_a_shared_sample() {
        let corpus = vec![words("she put money in the bank"), words("we sat on the river bank"), words("the bank was closed")];
        let sets = build(&[(1, words("bank depository"))], &corpus, &exact(&corpus), &base, &FakeScorer, 10, 2.0);
        assert_eq!(sets[0].samples.len(), 1, "only the money sentence");
        assert_eq!(sets[0].samples[0], SharedSample { sentence: 0, slot: 5, fits: words("bank depository") });
    }

    #[test]
    fn the_margin_decides_how_close_a_fit_must_be() {
        let s = words("she put money in the bank");
        assert_eq!(fitting_members(&FakeScorer, &s, 5, "bank", &words("bank depository"), &|l| l.to_string(), 0.1), words("bank"));
    }

    #[test]
    fn per_member_caps_the_sentences_tried() {
        let corpus: Vec<Vec<String>> = (0..20).map(|_| words("money in the bank")).collect();
        let sets = build(&[(1, words("bank depository"))], &corpus, &exact(&corpus), &base, &FakeScorer, 3, 2.0);
        assert_eq!(sets[0].samples.len(), 3);
    }

    /// Scores only the exact plural "obligations" as fitting "these ___".
    struct PluralScorer;
    impl SlotScorer for PluralScorer {
        fn slot_scores(&self, _: &[String], _: usize, candidates: &[String]) -> Option<Vec<f32>> {
            Some(candidates.iter().map(|c| if c == "duties" || c == "obligations" { -1.0 } else { -9.0 }).collect())
        }
    }

    #[test]
    fn other_members_are_tried_in_the_original_s_form() {
        let corpus = vec![words("these duties are heavy")];
        let occ = |_: u32, m: &str| if m == "duty" { vec![Occurrence { sentence: 0, slot: 1, xpos: "NNS" }] } else { vec![] };
        let plural = |lemma: &str, xpos: &str| if xpos == "NNS" { format!("{lemma}s") } else { lemma.to_string() };
        let sets = build(&[(7, words("duty obligation"))], &corpus, &occ, &plural, &PluralScorer, 10, 2.0);
        assert_eq!(sets[0].samples.len(), 1);
        assert_eq!(sets[0].samples[0].fits, words("duty obligation"), "fits are reported as lemmas");
    }

    #[test]
    fn junk_and_function_words_are_not_offered() {
        let stop = |w: &str| w == "the";
        assert!(usable_member("duty", &stop) && usable_member("well-being", &stop) && usable_member("o'clock", &stop));
        assert!(!usable_member("r", &stop), "single letters");
        assert!(!usable_member("the", &stop), "stoplist");
        assert!(!usable_member("3d", &stop) && !usable_member("c++", &stop), "non-words");
    }
}
