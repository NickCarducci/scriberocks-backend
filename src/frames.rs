//! Clause frames from the dependency tree: who wants / should / can't do
//! what, and how each clause links to the next. Every field traces back to
//! a word in the rock or a dictionary entry.

use crate::engine::{Engine, Sentence, CONTENT};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    Should,
    Must,
    Want,
    Can,
    Cannot,
    Will,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Link {
    Because,
    So,
    But,
    If,
    When,
    And,
}

#[derive(Debug, Serialize)]
pub struct Frame {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modality: Option<Modality>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub negated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verb: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    /// How this clause attaches to the clause it depends on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link: Option<Link>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub question: bool,
    /// Dictionary categories of the content words (WordNet lexicographer files).
    pub categories: Vec<String>,
    /// Abstract nouns the dictionary links to the words used: afraid → fear.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub concepts: Vec<String>,
}

/// Relations that start a clause of their own.
pub const CLAUSE_RELS: &[&str] = &["root", "advcl", "conj", "ccomp", "parataxis"];

/// Heads of the clauses in a sentence (root first).
pub fn clause_heads(s: &Sentence) -> Vec<usize> {
    let mut heads: Vec<usize> = s
        .tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| t.head.is_none() || (CLAUSE_RELS.contains(&t.deprel.as_str()) && matches!(t.upos.as_str(), "VERB" | "ADJ" | "AUX")))
        .map(|(i, _)| i)
        .collect();
    heads.sort_by_key(|&i| (s.tokens[i].head.is_some(), i));
    heads
}

/// The clause a token belongs to: nearest clause head at or above it.
pub fn clause_of(s: &Sentence, heads: &[usize], mut i: usize) -> usize {
    loop {
        if heads.contains(&i) {
            return i;
        }
        match s.tokens[i].head {
            Some(h) => i = h,
            None => return i,
        }
    }
}

impl Engine {
    pub fn frames(&self, input: &str) -> Vec<Frame> {
        let mut out = Vec::new();
        for s in self.parse(input) {
            let heads = clause_heads(&s);
            for &h in &heads {
                out.push(self.frame(&s, &heads, h));
            }
        }
        out
    }

    fn frame(&self, s: &Sentence, heads: &[usize], h: usize) -> Frame {
        let t = &s.tokens;
        let kids: Vec<usize> = s.children(h).collect();
        let lemma = |i: usize| t[i].lemma.to_lowercase();
        let rel = |i: usize| t[i].deprel.as_str();

        // "want / need / have to" + xcomp: the xcomp carries the action.
        let xcomp = kids.iter().copied().find(|&k| rel(k) == "xcomp" && t[k].upos == "VERB");
        let (mut modality, action) = match (lemma(h).as_str(), xcomp) {
            ("want" | "wish" | "hope" | "like" | "love", Some(x)) => (Some(Modality::Want), x),
            ("need" | "have" | "got", Some(x)) => (Some(Modality::Must), x),
            ("want" | "wish" | "hope", None) => (Some(Modality::Want), h),
            _ => (None, h),
        };
        let mut negated = false;
        let action_kids: Vec<usize> = s.children(action).collect();
        for &k in kids.iter().chain(action_kids.iter()) {
            match (rel(k), lemma(k).as_str()) {
                ("aux", "should" | "ought") => modality = Some(Modality::Should),
                ("aux", "must") => modality = Some(Modality::Must),
                ("aux", "can" | "could") => modality = modality.or(Some(Modality::Can)),
                ("aux", "will" | "shall") => modality = modality.or(Some(Modality::Will)),
                ("advmod", "not" | "never") => negated = true,
                _ => {}
            }
        }
        if modality == Some(Modality::Can) && negated {
            modality = Some(Modality::Cannot);
            negated = false;
        }

        let subject = kids
            .iter()
            .copied()
            .find(|&k| rel(k) == "nsubj" || rel(k) == "expl")
            .map(|k| if t[k].upos == "PRON" { t[k].form.to_lowercase() } else { lemma(k) });
        let verb = (t[action].upos == "VERB").then(|| lemma(action));
        let object = if t[action].upos == "VERB" {
            s.children(action).find(|&k| matches!(rel(k), "obj" | "obl" | "xcomp") && t[k].upos != "VERB").map(lemma)
        } else {
            Some(lemma(action)) // copular clause: "I'm tired" → tired
        };
        let link = kids.iter().copied().find(|&k| matches!(rel(k), "mark" | "cc")).and_then(|k| match lemma(k).as_str() {
            "because" | "since" | "cause" => Some(Link::Because),
            "so" => Some(Link::So),
            "but" | "though" | "although" | "yet" => Some(Link::But),
            "if" | "unless" => Some(Link::If),
            "when" | "while" | "whenever" => Some(Link::When),
            "and" => Some(Link::And),
            _ => None,
        });

        let members: Vec<usize> = (0..t.len()).filter(|&i| clause_of(s, heads, i) == h).collect();
        let mut categories = Vec::new();
        let mut concepts = Vec::new();
        for &i in &members {
            let Some(sense) = t[i].sense else { continue };
            if !CONTENT.contains(&t[i].upos.as_str()) || is_light(&lemma(i)) {
                continue;
            }
            let cat = self.lexicon.category_name(sense).to_string();
            if !matches!(cat.as_str(), "adj.all" | "adv.all" | "adj.ppl") && !categories.contains(&cat) {
                categories.push(cat);
            }
            for &a in &self.lexicon.synsets[sense as usize].attributes {
                let syn = &self.lexicon.synsets[a as usize];
                if let Some(name) = syn.members.first().filter(|_| syn.pos == crate::lexicon::Pos::Noun) {
                    if !concepts.contains(name) {
                        concepts.push(name.clone());
                    }
                }
            }
        }
        let mut words: Vec<&str> = members
            .iter()
            .filter(|&&i| !(matches!(rel(i), "punct" | "cc" | "mark") && i == members[0]))
            .map(|&i| t[i].form.as_str())
            .collect();
        while words.last().is_some_and(|w| matches!(*w, "," | ";")) {
            words.pop();
        }

        Frame {
            text: crate::text::detokenize(&words),
            subject,
            modality,
            negated,
            verb,
            object,
            link,
            question: s.ends_with_question(),
            categories,
            concepts,
        }
    }
}

/// Verbs too general to say anything about a clause's topic.
pub fn is_light(lemma: &str) -> bool {
    matches!(lemma, "be" | "have" | "do" | "get" | "want" | "need" | "go" | "make" | "will" | "can" | "should")
}
