//! The engine: every model loaded once, text in, analyzed sentences out.

use crate::inflect::Inflector;
use crate::lexicon::{Lexicon, Pos};
use crate::lm::LanguageModel;
use crate::parser::Parser;
use crate::tagger::Tagger;
use crate::text;
use crate::usage::model::{Device, UsageModel};
use crate::usage::senses::SenseIndex;
use crate::usage::sense_head::{Candidate, SenseHead};
use serde::Serialize;
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct Token {
    pub form: String,
    pub upos: String,
    pub xpos: String,
    pub lemma: String,
    /// Index of the head token in the sentence; None for the root.
    pub head: Option<usize>,
    pub deprel: String,
    /// Dictionary sense (synset index) for content words.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sense: Option<u32>,
    /// The dictionary supports that sense (single sense, or definition overlap).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub sense_supported: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Sentence {
    pub tokens: Vec<Token>,
}

impl Sentence {
    pub fn root(&self) -> Option<usize> {
        self.tokens.iter().position(|t| t.head.is_none())
    }

    pub fn children(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        self.tokens.iter().enumerate().filter(move |(_, t)| t.head == Some(i)).map(|(j, _)| j)
    }

    /// The token and everything below it, in sentence order.
    pub fn subtree(&self, i: usize) -> Vec<usize> {
        let mut out = vec![i];
        let mut k = 0;
        while k < out.len() {
            let node = out[k];
            out.extend(self.children(node));
            k += 1;
        }
        out.sort();
        out
    }

    pub fn ends_with_question(&self) -> bool {
        self.tokens.last().is_some_and(|t| t.form == "?")
    }
}

pub struct Engine {
    pub lexicon: Lexicon,
    pub upos: Tagger,
    pub xpos: Tagger,
    pub parser: Parser,
    pub inflector: Inflector,
    pub lm: LanguageModel,
    /// Optional: the usage model (fills in words from context) and its
    /// example-based sense index. Without them the engine falls back to
    /// dictionary overlap and fluency alone.
    pub usage: Option<UsageModel>,
    pub senses: Option<SenseIndex>,
    /// Optional learned sense head (`usage/head/`, from `scribe train-head`).
    /// When present it chooses senses instead of the example index.
    pub head: Option<SenseHead>,
    /// See `SENSE_MARGIN`; a field so experiments can vary it.
    pub sense_margin: f32,
    /// See `HEAD_MARGIN`.
    pub head_margin: f32,
    /// Optional neural syntax (`syntax/`, from examples/parse_finetune.rs):
    /// refines the perceptron's UPOS tags and the transition parser's arcs.
    pub syntax: Option<crate::syntax::SyntaxModel>,
    /// ScribeIE's enhancer, if one is loaded (used only when a request turns
    /// Generate on; see scribe_ie.rs).
    pub enhancer: Option<Box<dyn crate::scribe_ie::Enhancer>>,
    /// Replaces the usage model as the judge of swaps when set (e.g. a
    /// BERT-format model, to compare judges on the same rocks).
    pub judge: Option<Box<dyn SlotScorer + Send + Sync>>,
}

pub const CONTENT: &[&str] = &["NOUN", "VERB", "ADJ", "ADV"];

/// The boundary between the classical head start and a learned head: the
/// classical stages propose a closed set of candidates for a slot; a scorer
/// says how well each fits in context (log-probability). The usage model is
/// one implementation; a FidelityAI scoring head or a plain classical one can
/// implement the same trait and be compared with the same tests.
pub trait SlotScorer {
    /// `context[slot]` is replaced by each candidate (which may be several words).
    fn slot_scores(&self, context: &[String], slot: usize, candidates: &[String]) -> Option<Vec<f32>>;
}

impl SlotScorer for crate::bert::Bert {
    fn slot_scores(&self, context: &[String], slot: usize, candidates: &[String]) -> Option<Vec<f32>> {
        crate::bert::Bert::slot_scores(self, context, slot, candidates).ok()
    }
}

impl SlotScorer for UsageModel {
    fn slot_scores(&self, context: &[String], slot: usize, candidates: &[String]) -> Option<Vec<f32>> {
        self.candidate_scores(context, slot, candidates).ok()
    }
}

/// Usage evidence must beat the first-listed sense by this cosine margin to
/// override it. Lower margins score higher on the held-out dictionary
/// examples (45% exact at 0.05 vs 38% at 0.5) but that test spreads senses
/// evenly; on natural text (EWT) 0.05 overrides the first-listed sense 59%
/// of the time, often wrongly, while 0.5 overrides 4%. Retune on natural,
/// sense-tagged text (e.g. SemCor) — see examples/margin_sweep.rs.
pub const SENSE_MARGIN: f32 = 0.5;

/// The sense head must score another sense this far above the first-listed
/// one to override it. 3.0 is best on the ALL test set (Word Sense
/// Disambiguation Evaluation Framework): 61.0% with the head trained on all
/// of SemCor's training documents, against 55.6% for first-listed
/// (examples/semcor_eval.rs).
pub const HEAD_MARGIN: f32 = 3.0;

impl Engine {
    /// Loads lexicon, tagger, xpos, parser, inflect and lm `.bin` files from
    /// `dir`, plus `usage/` (usage model + sense index) when present.
    pub fn load(dir: &Path) -> Result<Engine, Box<dyn std::error::Error>> {
        let lexicon = Lexicon::load(&dir.join("lexicon.bin"))?;
        Ok(Engine {
            upos: Tagger::load(&dir.join("tagger.bin"))?,
            xpos: Tagger::load(&dir.join("xpos.bin"))?,
            parser: Parser::load(&dir.join("parser.bin"))?,
            inflector: Inflector::load(&dir.join("inflect.bin"))?,
            lm: LanguageModel::load(&dir.join("lm.bin"))?,
            usage: match dir.join("usage/usage.safetensors").exists() {
                true => Some(UsageModel::load(&dir.join("usage"), Device::Cpu)?),
                false => None,
            },
            senses: std::fs::read(dir.join("usage/senses.bin"))
                .ok()
                .and_then(|b| postcard::from_bytes::<SenseIndex>(&b).ok())
                .map(|mut index| {
                    index.build_backoff(&lexicon);
                    index
                }),
            head: match dir.join("usage/head/senses.json").exists() {
                true => Some(SenseHead::load(&dir.join("usage/head"), Device::Cpu)?),
                false => None,
            },
            syntax: match dir.join("syntax/parser_head.json").exists() {
                true => Some(crate::syntax::SyntaxModel::load(&dir.join("syntax"))?),
                false => None,
            },
            enhancer: None,
            judge: None,
            lexicon,
            sense_margin: SENSE_MARGIN,
            head_margin: HEAD_MARGIN,
        })
    }

    pub fn has_usage_model(&self) -> bool {
        self.usage.is_some()
    }

    /// The learned head used to judge closed-set candidates, if any.
    pub fn slot_scorer(&self) -> Option<&dyn SlotScorer> {
        match &self.judge {
            Some(j) => Some(j.as_ref() as &dyn SlotScorer),
            None => self.usage.as_ref().map(|u| u as &dyn SlotScorer),
        }
    }

    pub fn parse(&self, input: &str) -> Vec<Sentence> {
        text::sentences(input)
            .iter()
            .map(|s| text::tokenize(s))
            .filter(|t| !t.is_empty())
            .map(|tokens| self.parse_tokens(&tokens))
            .collect()
    }

    pub fn parse_tokens(&self, tokens: &[String]) -> Sentence {
        let words: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let upos = self.upos.tag(&words, Some(&self.lexicon));
        let xpos = self.xpos.tag(&words, Some(&self.lexicon));
        let arcs = self.parser.parse(&words, &upos);
        let (upos, arcs): (Vec<String>, _) = match self.syntax.as_ref().and_then(|m| m.refine(&words, &upos, &arcs).ok()) {
            Some(refined) => refined,
            None => (upos.iter().map(|t| t.to_string()).collect(), arcs),
        };
        let mut out: Vec<Token> = words
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let lemma = match Pos::from_upos(&upos[i]) {
                    Some(p) if upos[i] != "PROPN" => self.lexicon.lemmatize(w, Some(p)).into_iter().next(),
                    _ => None,
                }
                .unwrap_or_else(|| clitic_lemma(w));
                Token {
                    form: w.to_string(),
                    upos: upos[i].clone(),
                    xpos: xpos[i].to_string(),
                    lemma,
                    head: arcs[i].head.checked_sub(1),
                    deprel: arcs[i].label.clone(),
                    sense: None,
                    sense_supported: false,
                }
            })
            .collect();
        // Context vectors with each content word hidden (what the context says
        // the word means), one batched pass per sentence.
        let content: Vec<usize> = (0..out.len()).filter(|&i| CONTENT.contains(&out[i].upos.as_str()) && out[i].upos != "AUX").collect();
        let vectors: std::collections::HashMap<usize, Vec<f32>> = match &self.usage {
            Some(u) if !content.is_empty() && (self.senses.is_some() || self.head.is_some()) => {
                let lower: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
                let masked: Vec<Vec<String>> = content.iter().map(|&i| { let mut s = lower.clone(); s[i] = "[MASK]".into(); s }).collect();
                u.contextual_batch(&masked)
                    .map(|batch| content.iter().zip(batch).filter_map(|(&i, h)| h.get(i).cloned().map(|v| (i, v))).collect())
                    .unwrap_or_default()
            }
            _ => Default::default(),
        };
        for (i, t) in out.iter_mut().enumerate() {
            if CONTENT.contains(&t.upos.as_str()) && t.upos != "AUX" {
                if let Some(p) = Pos::from_upos(&t.upos) {
                    let v = vectors.get(&i).map(Vec::as_slice);
                    let senses = self.lexicon.senses(&t.lemma, p);
                    if let (Some(head), Some(v)) = (&self.head, v) {
                        let candidates: Vec<Candidate> =
                            senses.iter().enumerate().map(|(rank, &s)| Candidate { synset: s, category: self.lexicon.synsets[s as usize].category, rank }).collect();
                        if let Some((sense, supported)) = head.scores(v, &candidates).ok().and_then(|sc| choose_by_head(senses, &sc, self.head_margin, |s| head.knows(s))) {
                            t.sense = Some(sense);
                            t.sense_supported = supported;
                        }
                        continue;
                    }
                    let score = |s: u32| match (self.senses.as_ref(), v) {
                        (Some(index), Some(v)) => index.score(s, v, &self.lexicon),
                        _ => None,
                    };
                    if let Some((sense, supported)) = choose_sense(self.lexicon.senses(&t.lemma, p), score, self.sense_margin) {
                        t.sense = Some(sense);
                        t.sense_supported = supported;
                    }
                }
            }
        }
        Sentence { tokens: out }
    }

    pub fn fluency(&self, tokens: &[String]) -> f64 {
        let lower: Vec<String> = tokens.iter().map(|t| t.to_lowercase()).collect();
        self.lm.score(&lower)
    }
}

/// Chooses a sense from usage only — never from definitions. Default is the
/// first-listed sense (WordNet orders senses by frequency in tagged text);
/// another sense overrides it when its usage evidence matches this context
/// better by more than `margin`. `score` gives each sense's similarity and
/// evidence (own examples, or borrowed from relatives/category). The flag
/// says whether the chosen sense is backed by its own examples (or is the
/// word's only sense) — only then may swaps rely on it.
pub fn choose_sense(senses: &[u32], score: impl Fn(u32) -> Option<(f32, crate::usage::senses::Evidence)>, margin: f32) -> Option<(u32, bool)> {
    use crate::usage::senses::Evidence;
    let first = *senses.first()?;
    if senses.len() == 1 {
        return Some((first, true));
    }
    let first_score = score(first);
    let best = senses.iter().filter_map(|&s| score(s).map(|(x, e)| (s, x, e))).max_by(|a, b| a.1.total_cmp(&b.1));
    match best {
        Some((s, x, e)) if s != first && x - first_score.map_or(0.0, |f| f.0) > margin => Some((s, e == Evidence::Examples)),
        _ => Some((first, first_score.is_some_and(|f| f.1 == Evidence::Examples))),
    }
}

/// Chooses a sense from the sense head's scores (one per sense, in dictionary
/// order): the first-listed sense unless another scores more than `margin`
/// above it. The flag says whether the head trained on the chosen sense (or
/// it is the word's only sense) — only then may swaps rely on it.
pub fn choose_by_head(senses: &[u32], scores: &[f32], margin: f32, knows: impl Fn(u32) -> bool) -> Option<(u32, bool)> {
    let first = *senses.first()?;
    if senses.len() == 1 {
        return Some((first, true));
    }
    let best = (0..senses.len().min(scores.len())).max_by(|&a, &b| scores[a].total_cmp(&scores[b]))?;
    let chosen = if scores[best] - scores[0] > margin { senses[best] } else { first };
    Some((chosen, knows(chosen)))
}

fn clitic_lemma(w: &str) -> String {
    match w.to_lowercase().as_str() {
        "n't" => "not".into(),
        "'m" => "be".into(),
        "'re" => "be".into(),
        "'ll" => "will".into(),
        "'ve" => "have".into(),
        "'d" => "would".into(),
        "ca" => "can".into(),
        "wo" => "will".into(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod sense_tests {
    use super::*;

    #[test]
    fn the_head_overrides_the_first_listed_sense_only_past_its_margin() {
        let knows = |_: u32| true;
        assert_eq!(choose_by_head(&[10, 11], &[0.0, 4.0], 3.0, knows), Some((11, true)));
        assert_eq!(choose_by_head(&[10, 11], &[0.0, 2.0], 3.0, knows), Some((10, true)));
        assert_eq!(choose_by_head(&[10, 11, 12], &[1.0, 3.5, 4.5], 3.0, knows), Some((12, true)));
    }

    #[test]
    fn a_head_choice_is_supported_only_for_senses_the_head_trained_on() {
        let knows = |s: u32| s == 10;
        assert_eq!(choose_by_head(&[10, 11], &[0.0, 5.0], 3.0, knows), Some((11, false)));
        assert_eq!(choose_by_head(&[10, 11], &[0.0, 1.0], 3.0, knows), Some((10, true)));
        // A word's only sense is always supported; no senses, no choice.
        assert_eq!(choose_by_head(&[7], &[0.0], 3.0, |_| false), Some((7, true)));
        assert_eq!(choose_by_head(&[], &[], 3.0, |_| true), None);
    }

    use crate::usage::senses::Evidence;

    fn scores(table: &'static [(u32, f32, Evidence)]) -> impl Fn(u32) -> Option<(f32, Evidence)> {
        move |s| table.iter().find(|(x, _, _)| *x == s).map(|(_, v, e)| (*v, *e))
    }

    #[test]
    fn examples_override_the_first_listed_sense_only_past_the_margin() {
        let clear: &'static [(u32, f32, Evidence)] = &[(10, 0.0, Evidence::Examples), (11, 1.0, Evidence::Examples)];
        assert_eq!(choose_sense(&[10, 11], scores(clear), 0.5), Some((11, true)));
        let close: &'static [(u32, f32, Evidence)] = &[(10, 0.6, Evidence::Examples), (11, 0.8, Evidence::Examples)];
        assert_eq!(choose_sense(&[10, 11], scores(close), 0.5), Some((10, true)));
    }

    #[test]
    fn borrowed_evidence_can_pick_a_sense_but_never_supports_it() {
        // The right sense has no examples of its own; relatives make it reachable.
        let t: &'static [(u32, f32, Evidence)] = &[(10, 0.0, Evidence::Examples), (11, 0.9, Evidence::Relatives)];
        assert_eq!(choose_sense(&[10, 11], scores(t), 0.5), Some((11, false)));
    }

    #[test]
    fn falls_back_to_first_listed_sense_never_definitions() {
        assert_eq!(choose_sense(&[10, 11], |_| None, 0.5), Some((10, false)));
    }

    #[test]
    fn a_single_sense_is_always_supported() {
        assert_eq!(choose_sense(&[7], |_| None, 0.5), Some((7, true)));
        assert_eq!(choose_sense(&[], |_| None, 0.5), None);
    }
}
