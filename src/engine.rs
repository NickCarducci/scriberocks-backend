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

/// Weight of "listed first in the dictionary" when choosing senses by example.
pub const SENSE_ORDER_BONUS: f32 = 0.15;

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
        // Usage vectors for example-based senses, when the model is present.
        let vectors = match (&self.usage, &self.senses) {
            (Some(u), Some(_)) => u.contextual(&words.iter().map(|w| w.to_lowercase()).collect::<Vec<_>>()).ok(),
            _ => None,
        };
        for (i, t) in out.iter_mut().enumerate() {
            if CONTENT.contains(&t.upos.as_str()) && t.upos != "AUX" {
                if let Some(p) = Pos::from_upos(&t.upos) {
                    let v = vectors.as_ref().and_then(|v| v.get(i)).map(Vec::as_slice);
                    if let Some((sense, supported)) = choose_sense(self.lexicon.senses(&t.lemma, p), self.senses.as_ref(), v) {
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

/// Chooses a sense from usage only — never from definitions:
/// by example sentences when any of the word's senses have them (with a small
/// bonus for senses the dictionary lists first), otherwise the first-listed
/// sense (WordNet orders senses by how often they occur in tagged text).
/// The flag says whether usage supports the choice (examples, or only one sense).
pub fn choose_sense(senses: &[u32], index: Option<&SenseIndex>, v: Option<&[f32]>) -> Option<(u32, bool)> {
    let first = *senses.first()?;
    if senses.len() == 1 {
        return Some((first, true));
    }
    if let (Some(index), Some(v)) = (index, v) {
        if senses.iter().any(|&s| index.knows(s)) {
            return senses
                .iter()
                .enumerate()
                .map(|(rank, &s)| (s, index.similarity(s, v).unwrap_or(0.0) + SENSE_ORDER_BONUS / (1.0 + rank as f32)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(s, _)| (s, true));
        }
    }
    Some((first, false))
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
    fn picks_by_example_when_a_sense_has_examples() {
        let index = SenseIndex::from_vectors(vec![(10, vec![1.0, 0.0]), (11, vec![0.0, 1.0])]);
        assert_eq!(choose_sense(&[10, 11], Some(&index), Some(&[0.0, 1.0])), Some((11, true)));
    }

    #[test]
    fn falls_back_to_first_listed_sense_never_definitions() {
        assert_eq!(choose_sense(&[10, 11], None, None), Some((10, false)));
        let index = SenseIndex::from_vectors(vec![(99, vec![1.0])]);
        assert_eq!(choose_sense(&[10, 11], Some(&index), Some(&[1.0])), Some((10, false)));
    }

    #[test]
    fn a_single_sense_is_always_supported() {
        assert_eq!(choose_sense(&[7], None, None), Some((7, true)));
        assert_eq!(choose_sense(&[], None, None), None);
    }
}
