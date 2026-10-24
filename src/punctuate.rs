//! Punctuation, split the way the rest of the engine is: a classical
//! proposer (the parse says *where* punctuation could go) and a learned
//! scorer (the usage model says *whether* "," or "." fits there).
//! Closed set — no free generation.

use crate::engine::Sentence;

/// A mark is inserted when the scorer gives it at least this probability
/// at the gap (ln 0.15).
pub const MIN_MARK_LOGP: f32 = -1.9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GapKind {
    /// Before "so" / "but" / "yet" joining two clauses: a comma.
    Clause,
    /// After an opening word like "Honestly": a comma.
    Opener,
    /// Two clauses run together with nothing between them: comma or period.
    RunOn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gap {
    /// Punctuation would go before this token.
    pub before: usize,
    pub kind: GapKind,
}

const CLAUSE_JOINERS: &[&str] = &["so", "but", "yet"];
const OPENING_WORDS: &[&str] = &["honestly", "however", "well", "anyway", "actually", "basically", "still", "also", "yes", "no", "okay", "ok", "frankly", "luckily", "sadly", "hopefully"];
/// Verbs that take a clause as reported speech/thought ("I think I went"):
/// a bare clause after them is not a run-on.
const REPORTING: &[&str] = &[
    "think", "know", "say", "said", "feel", "guess", "believe", "hope", "wish", "mean", "realize", "suppose", "bet", "swear", "tell",
    "see", "notice", "find", "hear", "remember", "forget", "worry", "doubt", "wonder", "admit", "promise", "decide", "pretend", "imagine",
    "figure", "assume", "understand", "agree", "learn", "show", "prove", "explain", "claim", "insist", "notice", "sure", "glad", "afraid",
];
/// Missing apostrophes with only one reading ("its", "wont", "ill" are words).
const CONTRACTIONS: &[(&str, &str)] = &[
    ("dont", "don't"), ("cant", "can't"), ("im", "I'm"), ("ive", "I've"), ("didnt", "didn't"), ("doesnt", "doesn't"),
    ("isnt", "isn't"), ("wasnt", "wasn't"), ("werent", "weren't"), ("arent", "aren't"), ("couldnt", "couldn't"),
    ("shouldnt", "shouldn't"), ("wouldnt", "wouldn't"), ("havent", "haven't"), ("hasnt", "hasn't"), ("hadnt", "hadn't"),
    ("thats", "that's"), ("theres", "there's"), ("whats", "what's"), ("youre", "you're"), ("theyre", "they're"),
];

/// "i" → "I" and unambiguous missing apostrophes, keeping a leading capital.
pub fn fix_spelling(words: &[String]) -> Vec<String> {
    words
        .iter()
        .map(|w| {
            let lower = w.to_lowercase();
            if lower == "i" {
                return "I".to_string();
            }
            match CONTRACTIONS.iter().find(|(from, _)| *from == lower) {
                Some((_, to)) if w.chars().next().is_some_and(char::is_uppercase) && !to.starts_with('I') => {
                    let mut c = to.chars();
                    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
                }
                Some((_, to)) => to.to_string(),
                None => w.clone(),
            }
        })
        .collect()
}

const QUESTION_WORDS: &[&str] = &["what", "where", "why", "how", "who", "when", "which", "whose"];

/// Where punctuation could go, from the parse alone.
pub fn gaps(s: &Sentence) -> Vec<Gap> {
    let t = &s.tokens;
    let punct_before = |j: usize| j > 0 && t[j - 1].upos == "PUNCT";
    let mut out = Vec::new();

    for (j, tok) in t.iter().enumerate() {
        // , so / , but between clauses. Robust to scrambled parses: a mid-
        // sentence conjunction with a verb somewhere after it.
        if j > 0 && tok.deprel == "cc" && CLAUSE_JOINERS.contains(&tok.lemma.as_str()) && !punct_before(j) {
            let verb_after = t[j + 1..].iter().any(|x| matches!(x.upos.as_str(), "VERB" | "AUX"));
            if verb_after {
                out.push(Gap { before: j, kind: GapKind::Clause });
            }
        }
        // run-on: a clause that starts with no connective or punctuation —
        // parataxis/conj, or a bare ccomp under a word that doesn't report speech.
        let bare_ccomp = tok.deprel == "ccomp" && tok.head.is_some_and(|h| !REPORTING.contains(&t[h].lemma.to_lowercase().as_str()));
        if (matches!(tok.deprel.as_str(), "parataxis" | "conj") || bare_ccomp) && matches!(tok.upos.as_str(), "VERB" | "ADJ" | "AUX") {
            let first = s.subtree(j)[0];
            let opens_plainly = !matches!(t[first].deprel.as_str(), "cc" | "mark" | "punct");
            let has_subject = s.children(j).any(|k| t[k].deprel == "nsubj" || t[k].deprel == "expl");
            if first > 0 && opens_plainly && has_subject && !punct_before(first) {
                out.push(Gap { before: first, kind: GapKind::RunOn });
            }
        }
    }
    // Honestly, …
    if let Some(first) = t.first() {
        let opener = (OPENING_WORDS.contains(&first.lemma.as_str()) && matches!(first.deprel.as_str(), "advmod" | "discourse")) || first.upos == "INTJ";
        if opener && t.len() > 2 && t[1].upos != "PUNCT" {
            out.push(Gap { before: 1, kind: GapKind::Opener });
        }
    }
    out.sort_by_key(|g| g.before);
    out.dedup_by_key(|g| g.before);
    out
}

/// The end mark a sentence is missing, if any: "?" for questions.
pub fn end_mark(s: &Sentence) -> Option<&'static str> {
    let t = &s.tokens;
    if t.last().is_some_and(|x| matches!(x.form.as_str(), "." | "!" | "?")) {
        return None;
    }
    let first = t.first()?;
    let question = QUESTION_WORDS.contains(&first.lemma.to_lowercase().as_str()) || first.upos == "AUX";
    Some(if question { "?" } else { "." })
}

/// Words with marks inserted (`before` indices refer to `words`) and an
/// optional end mark. A word after an inserted period is capitalized.
pub fn insert(words: &[String], marks: &[(usize, &str)], end: Option<&str>) -> Vec<String> {
    let mut out = Vec::with_capacity(words.len() + marks.len() + 1);
    let mut capitalize_next = false;
    for (i, w) in words.iter().enumerate() {
        for (_, mark) in marks.iter().filter(|(b, _)| *b == i) {
            out.push(mark.to_string());
            capitalize_next = *mark == ".";
        }
        if capitalize_next {
            let mut c = w.chars();
            out.push(c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default());
            capitalize_next = false;
        } else {
            out.push(w.clone());
        }
    }
    if let Some(end) = end {
        out.push(end.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Sentence, Token};

    /// (form, upos, head (1-based, 0 = root), deprel)
    fn sent(spec: &[(&str, &str, usize, &str)]) -> Sentence {
        Sentence {
            tokens: spec
                .iter()
                .map(|(f, u, h, d)| Token {
                    form: f.to_string(),
                    upos: u.to_string(),
                    xpos: String::new(),
                    lemma: f.to_lowercase(),
                    head: h.checked_sub(1),
                    deprel: d.to_string(),
                    sense: None,
                    sense_supported: false,
                    sense_gap: None,
                })
                .collect(),
        }
    }

    #[test]
    fn proposes_a_comma_before_a_so_or_but_clause() {
        // I was tired so I went to bed
        let s = sent(&[("I", "PRON", 3, "nsubj"), ("was", "AUX", 3, "cop"), ("tired", "ADJ", 0, "root"), ("so", "CCONJ", 6, "cc"),
                       ("I", "PRON", 6, "nsubj"), ("went", "VERB", 3, "conj"), ("to", "ADP", 8, "case"), ("bed", "NOUN", 6, "obl")]);
        assert!(gaps(&s).contains(&Gap { before: 3, kind: GapKind::Clause }));
    }

    #[test]
    fn proposes_a_comma_after_an_opening_discourse_word() {
        // Honestly I need a break
        let s = sent(&[("Honestly", "ADV", 3, "advmod"), ("I", "PRON", 3, "nsubj"), ("need", "VERB", 0, "root"), ("a", "DET", 5, "det"), ("break", "NOUN", 3, "obj")]);
        assert!(gaps(&s).contains(&Gap { before: 1, kind: GapKind::Opener }));
    }

    #[test]
    fn proposes_a_break_in_a_run_on() {
        // I was tired I went to bed
        let s = sent(&[("I", "PRON", 3, "nsubj"), ("was", "AUX", 3, "cop"), ("tired", "ADJ", 0, "root"),
                       ("I", "PRON", 5, "nsubj"), ("went", "VERB", 3, "parataxis"), ("to", "ADP", 7, "case"), ("bed", "NOUN", 5, "obl")]);
        assert!(gaps(&s).contains(&Gap { before: 3, kind: GapKind::RunOn }));
    }

    #[test]
    fn treats_a_bare_clause_under_a_non_reporting_word_as_a_run_on() {
        // The parser's actual tree for "I was tired I went to bed" (ccomp).
        let s = sent(&[("I", "PRON", 3, "nsubj"), ("was", "AUX", 3, "cop"), ("tired", "ADJ", 0, "root"),
                       ("I", "PRON", 5, "nsubj"), ("went", "VERB", 3, "ccomp"), ("to", "ADP", 7, "case"), ("bed", "VERB", 5, "obl")]);
        assert!(gaps(&s).contains(&Gap { before: 3, kind: GapKind::RunOn }));
        // …but "I think I went" is reported speech, not a run-on.
        let s = sent(&[("I", "PRON", 2, "nsubj"), ("think", "VERB", 0, "root"), ("I", "PRON", 4, "nsubj"), ("went", "VERB", 2, "ccomp")]);
        assert!(gaps(&s).is_empty());
    }

    #[test]
    fn finds_but_clauses_even_when_the_parse_is_scrambled() {
        // The parser's actual tree for "I tried so hard but nothing changed".
        let s = sent(&[("I", "PRON", 2, "nsubj"), ("tried", "VERB", 0, "root"), ("so", "ADV", 4, "advmod"), ("hard", "ADV", 2, "advmod"),
                       ("but", "CCONJ", 6, "cc"), ("nothing", "PRON", 2, "conj"), ("changed", "VERB", 2, "xcomp")]);
        assert!(gaps(&s).contains(&Gap { before: 4, kind: GapKind::Clause }));
        // "everyone but me" has no verb after "but".
        let s = sent(&[("I", "PRON", 2, "nsubj"), ("called", "VERB", 0, "root"), ("everyone", "PRON", 2, "obj"), ("but", "CCONJ", 5, "cc"), ("me", "PRON", 3, "conj")]);
        assert!(gaps(&s).is_empty());
    }

    #[test]
    fn fixes_lowercase_i_and_unambiguous_contractions_only() {
        let words: Vec<String> = "i dont know if its wont im".split(' ').map(String::from).collect();
        assert_eq!(fix_spelling(&words), "I don't know if its wont I'm".split(' ').map(String::from).collect::<Vec<_>>());
        let words: Vec<String> = "Dont".split(' ').map(String::from).collect();
        assert_eq!(fix_spelling(&words), vec!["Don't"]);
    }

    #[test]
    fn proposes_the_missing_end_mark_and_knows_questions() {
        let statement = sent(&[("I", "PRON", 2, "nsubj"), ("slept", "VERB", 0, "root")]);
        assert_eq!(end_mark(&statement), Some("."));
        let question = sent(&[("where", "ADV", 3, "advmod"), ("are", "AUX", 3, "cop"), ("keys", "NOUN", 0, "root")]);
        assert_eq!(end_mark(&question), Some("?"));
        let done = sent(&[("I", "PRON", 2, "nsubj"), ("slept", "VERB", 0, "root"), (".", "PUNCT", 2, "punct")]);
        assert_eq!(end_mark(&done), None);
    }

    #[test]
    fn does_not_propose_where_punctuation_already_is() {
        let s = sent(&[("Honestly", "ADV", 4, "advmod"), (",", "PUNCT", 1, "punct"), ("I", "PRON", 4, "nsubj"), ("rest", "VERB", 0, "root")]);
        assert!(gaps(&s).is_empty());
    }

    #[test]
    fn inserts_marks_and_capitalizes_after_a_period() {
        let words: Vec<String> = "I was tired I went to bed".split(' ').map(String::from).collect();
        assert_eq!(insert(&words, &[(3, ".")], Some(".")).join(" "), "I was tired . I went to bed .");
        let words: Vec<String> = "honestly i need a break".split(' ').map(String::from).collect();
        assert_eq!(insert(&words, &[(1, ",")], Some(".")).join(" "), "honestly , i need a break .");
    }
}
