//! The engine: every model loaded once, text in, analyzed sentences out.

use crate::inflect::Inflector;
use crate::lexicon::{Lexicon, Pos};
use crate::lm::LanguageModel;
use crate::parser::Parser;
use crate::tagger::Tagger;
use crate::text;
use crate::usage::model::{Device, UsageModel};
use crate::usage::senses::SenseIndex;
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

impl SlotScorer for UsageModel {
    fn slot_scores(&self, context: &[String], slot: usize, candidates: &[String]) -> Option<Vec<f32>> {
        self.candidate_scores(context, slot, candidates).ok()
    }
}

/// Examples must beat the first-listed sense by this cosine margin to
/// override it (tuned on half the held-out dictionary examples).
pub const SENSE_MARGIN: f32 = 0.5;

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
            senses: std::fs::read(dir.join("usage/senses.bin")).ok().and_then(|b| postcard::from_bytes(&b).ok()),
            lexicon,
        })
    }

    pub fn has_usage_model(&self) -> bool {
        self.usage.is_some()
    }

    /// The learned head used to judge closed-set candidates, if any.
    pub fn slot_scorer(&self) -> Option<&dyn SlotScorer> {
        self.usage.as_ref().map(|u| u as &dyn SlotScorer)
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
        let mut out: Vec<Token> = words
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let lemma = match Pos::from_upos(upos[i]) {
                    Some(p) if upos[i] != "PROPN" => self.lexicon.lemmatize(w, Some(p)).into_iter().next(),
                    _ => None,
                }
                .unwrap_or_else(|| clitic_lemma(w));
                Token {
                    form: w.to_string(),
                    upos: upos[i].to_string(),
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
        let vectors: std::collections::HashMap<usize, Vec<f32>> = match (&self.usage, &self.senses) {
            (Some(u), Some(_)) if !content.is_empty() => {
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
                    if let Some((sense, supported)) = choose_sense(self.lexicon.senses(&t.lemma, p), self.senses.as_ref(), v, SENSE_MARGIN) {
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
/// a sense whose example sentences match this context better by more than
/// `margin` overrides it. The flag says whether usage supports the choice
/// (only one sense, or example sentences back it).
pub fn choose_sense(senses: &[u32], index: Option<&SenseIndex>, v: Option<&[f32]>, margin: f32) -> Option<(u32, bool)> {
    let first = *senses.first()?;
    if senses.len() == 1 {
        return Some((first, true));
    }
    let (Some(index), Some(v)) = (index, v) else { return Some((first, false)) };
    let first_sim = index.similarity(first, v);
    let best = senses.iter().filter_map(|&s| index.similarity(s, v).map(|x| (s, x))).max_by(|a, b| a.1.total_cmp(&b.1));
    match best {
        Some((s, x)) if s != first && x - first_sim.unwrap_or(0.0) > margin => Some((s, true)),
        _ => Some((first, index.knows(first))),
    }
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
    fn examples_override_the_first_listed_sense_only_past_the_margin() {
        let index = SenseIndex::from_vectors(vec![(10, vec![1.0, 0.0]), (11, vec![0.0, 1.0])]);
        // Clearly sense 11 (cos 1 vs 0): margin 1.0 > 0.5.
        assert_eq!(choose_sense(&[10, 11], Some(&index), Some(&[0.0, 1.0]), 0.5), Some((11, true)));
        // Only slightly closer to 11: stays with the first-listed sense, which examples still support.
        assert_eq!(choose_sense(&[10, 11], Some(&index), Some(&[0.6, 0.8]), 0.5), Some((10, true)));
    }

    #[test]
    fn falls_back_to_first_listed_sense_never_definitions() {
        assert_eq!(choose_sense(&[10, 11], None, None, 0.5), Some((10, false)));
        let index = SenseIndex::from_vectors(vec![(99, vec![1.0])]);
        assert_eq!(choose_sense(&[10, 11], Some(&index), Some(&[1.0]), 0.5), Some((10, false)));
    }

    #[test]
    fn a_single_sense_is_always_supported() {
        assert_eq!(choose_sense(&[7], None, None, 0.5), Some((7, true)));
        assert_eq!(choose_sense(&[], None, None, 0.5), None);
    }
}
