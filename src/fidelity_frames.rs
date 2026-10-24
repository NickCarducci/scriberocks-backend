//! FidelityAI's path B grammar frames (extractor "scribe-v1"), in Rust: the
//! same rules as FidelityAI's scribe_parse.js `extractGrammarFrames`, over the
//! same tagger and parser (pruned at 1.0, as exported for FidelityAI), so a
//! Swift or Rust app can read text into a FidelityAI prose model's role values
//! without JavaScript. Kept identical to the JavaScript by a shared fixture
//! (tests/fixtures/fidelity_frames.jsonl, written by FidelityAI).
//!
//! Slots: subject, modal, action, object, target, modifier, condition, status.
use crate::engine::Engine;
use crate::lexicon::Pos;
use crate::text;
use serde::Serialize;

pub const EXTRACTOR: &str = "scribe-v1";
/// The pruning FidelityAI's exported model uses (examples/export_parse_model.rs).
pub const PRUNE: f32 = 1.0;

const MODALS: [&str; 10] = ["can", "could", "may", "might", "must", "shall", "should", "will", "would", "ought"];
const NEGATIONS: [&str; 4] = ["not", "n't", "never", "no"];
const TARGET_PREPS: [&str; 12] = ["to", "for", "from", "into", "onto", "with", "against", "on", "in", "through", "via", "by"];
const COND_WORDS: [&str; 6] = ["if", "when", "unless", "until", "before", "after"];
const NEGATIVE_RESULT_WORDS: [&str; 12] = ["abort", "block", "cancel", "deny", "error", "exception", "fail", "failure", "forbid", "reject", "terminate", "violate"];
const POSITIVE_RESULT_WORDS: [&str; 10] = ["accept", "allow", "complete", "confirm", "enable", "permit", "provide", "return", "succeed", "success"];

#[derive(Clone, Debug, Default, PartialEq, Serialize, serde::Deserialize)]
pub struct GrammarFrame {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
    pub status: String,
}

/// An engine whose tagger and parser are pruned as FidelityAI's model is.
pub fn prepare(engine: &mut Engine) {
    engine.upos.prune(PRUNE);
    engine.parser.prune(PRUNE);
}

/// Grammar frames for `input`: one per clause (the root, and clauses
/// coordinated with it). Call `prepare` on the engine first.
pub fn grammar_frames(engine: &Engine, input: &str) -> Vec<GrammarFrame> {
    let mut frames = Vec::new();
    for sentence in text::sentences(input) {
        frames.extend(sentence_frames(engine, &text::tokenize(&sentence)).into_iter().map(|(f, _)| f));
    }
    frames
}

/// The word (index into the sentence's tokens) filling each slot of a frame:
/// the head of the subject, object and target phrases, and the action's verb
/// (the copula for "be").
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SlotWords {
    pub subject: Option<usize>,
    pub action: Option<usize>,
    pub object: Option<usize>,
    pub target: Option<usize>,
}

/// Frames for one tokenized sentence, each with the words filling its slots.
pub fn sentence_frames(engine: &Engine, words: &[String]) -> Vec<(GrammarFrame, SlotWords)> {
    let mut frames = Vec::new();
    {
        if words.is_empty() {
            return frames;
        }
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let tags = engine.upos.tag(&refs, Some(&engine.lexicon));
        let arcs = engine.parser.parse(&refs, &tags);
        let base = |j: usize| arcs[j].label.split(':').next().unwrap_or("").to_string();
        let kids = |i: usize| -> Vec<usize> { (0..arcs.len()).filter(|&j| arcs[j].head == i + 1).collect() };
        let lower = |j: usize| words[j].to_lowercase();
        let lemma_as = |j: usize, pos: Option<Pos>| -> String {
            if tags[j] == "PRON" || tags[j] == "PROPN" {
                lower(j)
            } else {
                engine.lexicon.lemmatize(&words[j], pos).into_iter().next().unwrap_or_else(|| lower(j))
            }
        };
        let noun = |j: usize| -> String {
            let mut parts: Vec<String> = kids(j).into_iter().filter(|&k| base(k) == "compound" && k < j).map(|k| lemma_as(k, Some(Pos::Noun))).collect();
            parts.push(lemma_as(j, Some(Pos::Noun)));
            parts.join(" ")
        };
        let has_kid = |j: usize, rel: &str, set: &[&str]| kids(j).into_iter().any(|c| base(c) == rel && set.contains(&lower(c).as_str()));
        let clause = |h: usize| -> Option<(GrammarFrame, SlotWords)> {
            let ks = kids(h);
            let cop = ks.iter().copied().find(|&k| base(k) == "cop");
            let action = if tags[h] == "VERB" {
                Some(lemma_as(h, Some(Pos::Verb)))
            } else if cop.is_some() {
                Some("be".to_string())
            } else if tags[h] == "AUX" {
                Some(lemma_as(h, Some(Pos::Verb)))
            } else {
                None
            };
            let subj = ks.iter().copied().find(|&k| base(k) == "nsubj" || base(k) == "expl");
            let mut obj = ks.iter().copied().find(|&k| base(k) == "obj");
            let modal = ks.iter().copied().filter(|&k| base(k) == "aux").map(lower).find(|w| MODALS.contains(&w.as_str()));
            let negated = ks.iter().any(|&k| (base(k) == "advmod" || base(k) == "det") && NEGATIONS.contains(&lower(k).as_str()))
                || obj.is_some_and(|o| has_kid(o, "det", &NEGATIONS));
            let obl = ks.iter().copied().find(|&k| base(k) == "obl" && has_kid(k, "case", &TARGET_PREPS));
            let adv = ks.iter().copied().find(|&k| base(k) == "advmod" && !NEGATIONS.contains(&lower(k).as_str()));
            let cond = ks.iter().copied().find(|&k| base(k) == "advcl" && has_kid(k, "mark", &COND_WORDS));
            let mut f = GrammarFrame { subject: subj.map(noun), modal, action, ..Default::default() };
            // A copula's noun complement is what the clause is about: "X is a market".
            if obj.is_none() && cop.is_some() && (tags[h] == "NOUN" || tags[h] == "PROPN") {
                obj = Some(h);
            }
            f.object = obj.map(noun);
            f.target = obl.map(noun);
            f.modifier = match adv {
                Some(a) => Some(lemma_as(a, Some(Pos::Adv))),
                None if cop.is_some() && tags[h] == "ADJ" => Some(lemma_as(h, Some(Pos::Adj))),
                None => None,
            };
            f.condition = cond.map(|c| if tags[c] == "VERB" { lemma_as(c, Some(Pos::Verb)) } else { lemma_as(c, None) });
            let mut terms: Vec<String> = f.action.iter().cloned().collect();
            if let Some(o) = &f.object {
                terms.extend(o.split(' ').map(String::from));
            }
            f.status = if negated {
                "negative"
            } else if terms.iter().any(|t| NEGATIVE_RESULT_WORDS.contains(&t.as_str())) {
                "negative_result"
            } else if terms.iter().any(|t| POSITIVE_RESULT_WORDS.contains(&t.as_str())) {
                "affirmative_result"
            } else {
                "affirmative"
            }
            .to_string();
            let slots = SlotWords {
                subject: subj,
                action: if tags[h] == "VERB" || tags[h] == "AUX" { Some(h) } else { cop },
                object: obj,
                target: obl,
            };
            (f.action.is_some() || f.object.is_some()).then_some((f, slots))
        };
        // The clause head: the root when it's a predicate; otherwise (the parser
        // rooted a noun) the first verb with a subject of its own, else the first verb.
        let predicate = |j: usize| tags[j] == "VERB" || tags[j] == "AUX" || kids(j).into_iter().any(|c| base(c) == "cop");
        let mut root = arcs.iter().position(|a| a.head == 0);
        if root.is_none_or(|r| !predicate(r)) {
            let verbs: Vec<usize> = (0..tags.len()).filter(|&j| tags[j] == "VERB").collect();
            root = verbs.iter().copied().find(|&j| kids(j).into_iter().any(|k| base(k) == "nsubj")).or(verbs.first().copied()).or(root);
        }
        let Some(root) = root else { return frames };
        let mut heads = vec![root];
        heads.extend(kids(root).into_iter().filter(|&k| {
            (base(k) == "conj" || base(k) == "parataxis") && (tags[k] == "VERB" || kids(k).into_iter().any(|c| base(c) == "cop"))
        }));
        frames.extend(heads.into_iter().filter_map(clause));
    }
    frames
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every fixture text gets exactly the frames FidelityAI's JavaScript gives
    /// (fixture from FidelityAI engine/scripts/frames_fixture.mjs; texts from
    /// the UD English Web Treebank test split, CC BY-SA 4.0, plus edge cases).
    #[test]
    fn frames_match_fidelityai_javascript() {
        let Ok(mut e) = Engine::load(std::path::Path::new("models")) else { return };
        prepare(&mut e);
        #[derive(serde::Deserialize)]
        struct Case { text: String, frames: Vec<GrammarFrame> }
        let fixture = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fidelity_frames.jsonl")).unwrap();
        let mut n = 0;
        for line in fixture.lines().filter(|l| !l.is_empty()) {
            let case: Case = serde_json::from_str(line).unwrap();
            assert_eq!(grammar_frames(&e, &case.text), case.frames, "{}", case.text);
            n += 1;
        }
        eprintln!("compared {n} fixture texts");
        assert!(n >= 200);
    }
}
