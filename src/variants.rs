//! Variants of a rock, built only from its own words plus the dictionary.
//!
//! Operations (each a mechanism; which ones to surface is a product choice):
//!   trim    — drop hedges ("really", "kind of", "I think that …")
//!   cut     — keep one clause, or drop one clause / sentence
//!   reorder — move a because/if/when clause to the other side
//!   swap    — replace a word with a dictionary synonym of its chosen sense
//!   reframe — hold the same action with a different modality (should → want to → will …)
//!   connect — join two sentences with a closed-set connective (because / so / but / and)
//!             when the usage model finds it as natural as the sentence break

use crate::engine::{Engine, Sentence, CONTENT};
use crate::frames::{clause_heads, clause_of, is_light};
use serde::Serialize;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    Trim,
    Cut,
    Reorder,
    Swap,
    Reframe,
    Connect,
}

#[derive(Clone, Debug, Serialize)]
pub struct Variant {
    pub text: String,
    pub ops: Vec<Op>,
    /// Change in fluency vs the original (mean log-prob per token; 0 = same).
    pub fluency_delta: f64,
    /// Byte ranges in `text` of words that aren't in the original.
    pub changed: Vec<(usize, usize)>,
}

struct Candidate {
    tokens: Vec<String>,
    ops: Vec<Op>,
}

const HEDGES: &[&str] = &["just", "really", "actually", "basically", "literally", "honestly", "totally", "kinda", "sorta", "definitely", "seriously"];
const OPINION_VERBS: &[&str] = &["think", "guess", "feel", "believe", "suppose", "reckon"];
const MOVABLE_MARKS: &[&str] = &["because", "if", "when", "since", "although", "though", "once", "unless", "while"];
const MODALS: &[&str] = &["should", "must", "can", "could", "will", "would", "might", "shall", "may"];
/// Degree and quantity words: grammatical glue, not content to swap.
const NO_SWAP: &[&str] = &[
    "more", "most", "less", "least", "much", "many", "few", "very", "so", "too", "enough", "quite", "rather", "also", "then", "now", "here",
    "there", "again", "still", "even", "only", "ever", "never", "always", "back", "away", "out", "up", "down", "off", "on", "well", "how",
    "where", "when", "why", "not", "lot", "thing", "way", "time",
];

/// Below this fluency change a variant is treated as broken and dropped.
pub const MIN_FLUENCY_DELTA: f64 = -2.5;

/// A swap must fit the sentence (usage-model log-probability) within this
/// margin of the original word, and at most this many are kept per word.
pub const SWAP_MARGIN: f32 = 2.5;
pub const SWAPS_PER_WORD: usize = 3;

/// A connective is kept if it scores within this margin of a plain sentence
/// break at the same spot, at most `CONNECTS_PER_JOIN` per sentence pair.
pub const CONNECT_MARGIN: f32 = 1.0;
pub const CONNECTS_PER_JOIN: usize = 2;

/// Closed set for the connect operation: slot-filling, not free generation.
pub const CONNECTIVES: [&str; 4] = ["because", "so", "but", "and"];
/// Words that already open a linked sentence.
const OPENERS: &[&str] = &["because", "but", "so", "and", "or", "then", "also", "however", "if", "when", "since", "though", "although", "yet"];

/// Two adjacent sentences can be joined when both are plain statements and
/// the second doesn't already open with a connective.
pub fn can_join(first: &[String], second: &[String]) -> bool {
    fn ends(s: &[String]) -> &str {
        s.last().map(String::as_str).unwrap_or("")
    }
    let opener = second.first().map(|w| w.to_lowercase()).unwrap_or_default();
    first.len() >= 3 && second.len() >= 3 && matches!(ends(first), "." | "!") && ends(second) != "?" && !OPENERS.contains(&opener.as_str())
}

/// "A." + "B." → "A because b." ("so"/"but" take a comma). `keep_case` keeps
/// the second sentence's first word capitalized (names); "I" always is.
pub fn join_with(first: &[String], second: &[String], connective: &str, keep_case: bool) -> Vec<String> {
    let mut out: Vec<String> = first.to_vec();
    while out.last().is_some_and(|w| matches!(w.as_str(), "." | "!" | "?")) {
        out.pop();
    }
    if matches!(connective, "so" | "but") {
        out.push(",".into());
    }
    out.push(connective.to_string());
    for (i, w) in second.iter().enumerate() {
        out.push(if i == 0 && !keep_case && w != "I" && !w.starts_with("I'") { lower_first(w) } else { w.clone() });
    }
    out
}

/// Swaps scoring within `margin` of the original, best first, at most `max`.
pub fn keep_fitting(original: f32, mut scored: Vec<(String, f32)>, margin: f32, max: usize) -> Vec<String> {
    scored.retain(|(_, s)| *s >= original - margin);
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.into_iter().take(max).map(|(w, _)| w).collect()
}

impl Engine {
    /// Up to `limit` variants of `rock`, best first, adjacent ones kept dissimilar.
    pub fn variants(&self, rock: &str, limit: usize) -> Vec<Variant> {
        let sentences = self.parse(rock);
        if sentences.is_empty() {
            return vec![];
        }
        let originals: Vec<Vec<String>> = sentences.iter().map(|s| s.tokens.iter().map(|t| t.form.clone()).collect()).collect();
        let original_text = join_sentences(&originals);
        let all_original: Vec<String> = originals.concat();

        let mut seen: HashSet<String> = HashSet::from([original_text.to_lowercase()]);
        let mut out: Vec<Variant> = Vec::new();
        let mut push = |sents: Vec<Vec<String>>, ops: Vec<Op>, delta: f64, out: &mut Vec<Variant>| {
            let (text, offsets) = join_with_offsets(&sents);
            if delta < MIN_FLUENCY_DELTA || !seen.insert(text.to_lowercase()) {
                return;
            }
            let changed = changed_ranges(&all_original, &sents.concat(), &offsets);
            out.push(Variant { text, ops, fluency_delta: delta, changed });
        };

        // Whole-sentence cuts for multi-sentence rocks.
        if originals.len() > 1 {
            for i in 0..originals.len() {
                push(vec![originals[i].clone()], vec![Op::Cut], 0.0, &mut out);
                if originals.len() > 2 {
                    let rest: Vec<Vec<String>> = originals.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, s)| s.clone()).collect();
                    push(rest, vec![Op::Cut], 0.0, &mut out);
                }
            }
        }

        // Joins between adjacent sentences (needs the usage model to judge fit).
        for (i, connective) in self.connections(&originals) {
            let keep_case = sentences[i + 1].tokens.first().is_some_and(|t| t.upos == "PROPN");
            let joined = join_with(&originals[i], &originals[i + 1], connective, keep_case);
            let base = self.fluency(&[originals[i].clone(), originals[i + 1].clone()].concat());
            let delta = self.fluency(&joined) - base;
            let mut sents = originals.clone();
            sents.splice(i..=i + 1, [joined]);
            push(sents, vec![Op::Connect], delta, &mut out);
        }

        for (si, s) in sentences.iter().enumerate() {
            let base = self.fluency(&originals[si]);
            for c in self.sentence_candidates(s) {
                let delta = self.fluency(&c.tokens) - base;
                let mut sents = originals.clone();
                sents[si] = c.tokens;
                push(sents, c.ops, delta, &mut out);
            }
        }

        diversify(out, limit)
    }

    /// (sentence index, connective) pairs where joining sentence i and i+1
    /// reads about as naturally as the existing break.
    fn connections(&self, originals: &[Vec<String>]) -> Vec<(usize, &'static str)> {
        let Some(usage) = &self.usage else { return vec![] };
        let mut out = Vec::new();
        for i in 0..originals.len().saturating_sub(1) {
            let (a, b) = (&originals[i], &originals[i + 1]);
            if !can_join(a, b) {
                continue;
            }
            // "a … [slot] b …": score each connective and "." in the slot.
            let mut context: Vec<String> = a[..a.len() - 1].iter().map(|w| w.to_lowercase()).collect();
            let slot = context.len();
            context.push("[MASK]".into());
            context.extend(b.iter().map(|w| w.to_lowercase())); // the usage model is uncased
            let mut candidates: Vec<String> = CONNECTIVES.iter().map(|c| c.to_string()).collect();
            candidates.push(".".into());
            let Ok(scores) = usage.candidate_scores(&context, slot, &candidates) else { continue };
            let sentence_break = scores[CONNECTIVES.len()];
            let scored: Vec<(String, f32)> = CONNECTIVES.iter().map(|c| c.to_string()).zip(scores.iter().copied()).collect();
            for c in keep_fitting(sentence_break, scored, CONNECT_MARGIN, CONNECTS_PER_JOIN) {
                let c: &'static str = CONNECTIVES.iter().find(|x| **x == c).unwrap();
                out.push((i, c));
            }
        }
        out
    }

    fn sentence_candidates(&self, s: &Sentence) -> Vec<Candidate> {
        let all: Vec<usize> = (0..s.tokens.len()).collect();
        // Trimmed view first, so a result reachable both ways is labeled "trim".
        let mut views = Vec::new();
        if let Some(trimmed) = trim(s) {
            views.push((trimmed, vec![Op::Trim]));
        }
        views.push((all, vec![]));
        let mut out = Vec::new();
        for (kept, ops) in views {
            let with = |op: Op| ops.iter().copied().chain([op]).collect::<Vec<_>>();
            if !ops.is_empty() {
                out.push(Candidate { tokens: words(s, &kept), ops: ops.clone() });
            }
            for cut in cuts(s, &kept) {
                out.push(Candidate { tokens: words(s, &cut), ops: with(Op::Cut) });
            }
            if let Some(tokens) = reorder(s, &kept) {
                out.push(Candidate { tokens, ops: with(Op::Reorder) });
            }
            for tokens in self.swaps(s, &kept) {
                out.push(Candidate { tokens, ops: with(Op::Swap) });
            }
            for tokens in reframes(s, &kept) {
                out.push(Candidate { tokens, ops: with(Op::Reframe) });
            }
        }
        out
    }

    fn swaps(&self, s: &Sentence, kept: &[usize]) -> Vec<Vec<String>> {
        let mut out = Vec::new();
        for (pos, &i) in kept.iter().enumerate() {
            let t = &s.tokens[i];
            let lemma = t.lemma.to_lowercase();
            if !CONTENT.contains(&t.upos.as_str()) || t.upos == "AUX" || is_light(&lemma) || HEDGES.contains(&lemma.as_str()) || NO_SWAP.contains(&lemma.as_str()) {
                continue;
            }
            let Some(sense) = t.sense.filter(|_| t.sense_supported) else { continue };
            let synset = &self.lexicon.synsets[sense as usize];
            // Adjectives: also the dictionary's cluster of similar adjectives.
            let similar = if t.upos == "ADJ" { synset.similar.as_slice() } else { &[] };
            let candidates = synset.members.iter().chain(similar.iter().flat_map(|&x| self.lexicon.synsets[x as usize].members.iter()));
            let neighbors: Vec<String> = [pos.checked_sub(1), Some(pos + 1)].into_iter().flatten().filter_map(|p| kept.get(p)).map(|&j| s.tokens[j].lemma.to_lowercase()).collect();
            let modifier = matches!(t.deprel.as_str(), "advmod" | "amod");
            let mut seen = HashSet::new();
            let mut forms: Vec<String> = Vec::new();
            for member in candidates {
                let m = member.to_lowercase();
                if m == lemma || !seen.insert(m.clone()) || m.chars().any(|c| c.is_ascii_digit()) || m.split(' ').count() > 3 {
                    continue;
                }
                if neighbors.contains(&m) || (modifier && m.contains(' ')) {
                    continue; // "go go-ing", "to a fault tired"
                }
                let form = match t.upos.as_str() {
                    "VERB" | "NOUN" | "ADJ" | "ADV" => self.inflector.inflect(&m, &t.xpos),
                    _ => m.clone(),
                };
                let form = if t.form.chars().next().is_some_and(char::is_uppercase) { capitalize(&form) } else { form };
                forms.push(form);
            }
            // With the usage model, keep only words that fit this sentence.
            if let Some(usage) = &self.usage {
                let context: Vec<String> = words(s, kept).iter().map(|w| w.to_lowercase()).collect();
                let mut all = vec![t.form.to_lowercase()];
                all.extend(forms.iter().map(|f| f.to_lowercase()));
                if let Ok(scores) = usage.candidate_scores(&context, pos, &all) {
                    let fitting = keep_fitting(scores[0], forms.iter().cloned().zip(scores[1..].iter().copied()).collect(), SWAP_MARGIN, SWAPS_PER_WORD);
                    forms.retain(|f| fitting.contains(f));
                }
            }
            for form in forms {
                let mut tokens = words(s, kept);
                tokens.splice(pos..pos + 1, form.split(' ').map(String::from));
                fix_article(&mut tokens, pos);
                out.push(tokens);
            }
        }
        out
    }
}

/// Indices to keep after removing hedges, or None if there are none.
fn trim(s: &Sentence) -> Option<Vec<usize>> {
    let t = &s.tokens;
    let mut drop: HashSet<usize> = HashSet::new();
    for (i, tok) in t.iter().enumerate() {
        let l = tok.lemma.to_lowercase();
        if HEDGES.contains(&l.as_str()) && matches!(tok.deprel.as_str(), "advmod" | "discourse") {
            drop.insert(i);
        }
        if matches!(l.as_str(), "kind" | "sort")
            && t.get(i + 1).is_some_and(|n| n.form.eq_ignore_ascii_case("of"))
            && t.get(i + 2).is_some_and(|n| matches!(n.upos.as_str(), "ADJ" | "VERB" | "ADV"))
        {
            drop.extend([i, i + 1]);
        }
    }
    // "I think (that) X" → X
    if let Some(r) = s.root() {
        let is_opinion = OPINION_VERBS.contains(&t[r].lemma.as_str()) && s.children(r).any(|k| t[k].deprel == "nsubj" && t[k].form.eq_ignore_ascii_case("i"));
        if let Some(c) = is_opinion.then(|| s.children(r).find(|&k| t[k].deprel == "ccomp")).flatten() {
            let keep: HashSet<usize> = s.subtree(c).into_iter().filter(|&k| !(t[k].deprel == "mark" && t[k].form.eq_ignore_ascii_case("that"))).collect();
            drop.extend((0..t.len()).filter(|k| !keep.contains(k) && !(t[*k].deprel == "punct" && *k == t.len() - 1)));
        }
    }
    // A comma left behind a dropped hedge goes too: "Honestly, I …" → "I …"
    for i in drop.clone() {
        if t.get(i + 1).is_some_and(|n| n.form == ",") && (i == 0 || drop.contains(&(i - 1)) || i + 1 == 1) {
            drop.insert(i + 1);
        }
    }
    if drop.is_empty() {
        return None;
    }
    let kept: Vec<usize> = (0..t.len()).filter(|i| !drop.contains(i)).collect();
    (kept.iter().any(|&i| CONTENT.contains(&t[i].upos.as_str()))).then_some(kept)
}

/// Each clause alone; with three or more clauses, also each clause dropped.
fn cuts(s: &Sentence, kept: &[usize]) -> Vec<Vec<usize>> {
    let t = &s.tokens;
    let heads: Vec<usize> = clause_heads(s).into_iter().filter(|h| kept.contains(h)).collect();
    if heads.len() < 2 {
        return vec![];
    }
    let unit = |h: usize| -> Vec<usize> {
        kept.iter().copied().filter(|&i| clause_of(s, &heads, i) == h).filter(|&i| !(matches!(t[i].deprel.as_str(), "punct") && i != t.len() - 1)).collect()
    };
    let connective = |i: usize| matches!(t[i].deprel.as_str(), "mark" | "cc") || (t[i].deprel == "advmod" && t[i].lemma == "so");
    let mut out = Vec::new();
    for &h in &heads {
        let mut alone: Vec<usize> = unit(h);
        while alone.first().is_some_and(|&i| connective(i) || t[i].deprel == "punct") {
            alone.remove(0);
        }
        let has_subject = alone.iter().any(|&i| t[i].deprel == "nsubj" || t[i].deprel == "expl");
        // "I think" without what's thought is a fragment.
        let complete = !s.children(h).any(|k| t[k].deprel == "ccomp" && !alone.contains(&k));
        if has_subject && complete && alone.len() >= 2 {
            out.push(alone);
        }
        if heads.len() >= 3 && t[h].head.is_some() {
            let gone: HashSet<usize> = unit(h).into_iter().collect();
            let rest: Vec<usize> = kept.iter().copied().filter(|i| !gone.contains(i)).collect();
            out.push(rest);
        }
    }
    out
}

/// "A because B." ⇄ "Because B, A."
fn reorder(s: &Sentence, kept: &[usize]) -> Option<Vec<String>> {
    let t = &s.tokens;
    let root = s.root()?;
    let a = s.children(root).find(|&k| {
        t[k].deprel == "advcl" && kept.contains(&k) && s.children(k).any(|m| t[m].deprel == "mark" && MOVABLE_MARKS.contains(&t[m].lemma.to_lowercase().as_str()))
    })?;
    let sub: HashSet<usize> = s.subtree(a).into_iter().collect();
    let end_punct = kept.last().filter(|&&i| t[i].deprel == "punct" && matches!(t[i].form.as_str(), "." | "!" | "?")).copied();
    let clause: Vec<usize> = kept.iter().copied().filter(|i| sub.contains(i) && Some(*i) != end_punct).collect();
    let mut main: Vec<usize> = kept.iter().copied().filter(|i| !sub.contains(i) && Some(*i) != end_punct).collect();
    while main.first().is_some_and(|&i| t[i].form == ",") {
        main.remove(0);
    }
    while main.last().is_some_and(|&i| t[i].form == ",") {
        main.pop();
    }
    let mut clause_words: Vec<String> = clause.iter().map(|&i| t[i].form.clone()).collect();
    while clause_words.last().is_some_and(|w| w == ",") {
        clause_words.pop();
    }
    let main_words: Vec<String> = main.iter().map(|&i| t[i].form.clone()).collect();
    if clause_words.is_empty() || main_words.is_empty() {
        return None;
    }
    let clause_first = clause.first().is_some_and(|&c| main.first().is_some_and(|&m| c < m));
    let mut out = Vec::new();
    if clause_first {
        // "Because B, A." → "A because B."
        out.extend(main_words.iter().enumerate().map(|(i, w)| if i == 0 { capitalize(w) } else { w.clone() }));
        out.extend(clause_words.iter().enumerate().map(|(i, w)| if i == 0 { lower_first(w) } else { w.clone() }));
    } else {
        out.extend(clause_words.iter().enumerate().map(|(i, w)| if i == 0 { capitalize(w) } else { w.clone() }));
        out.push(",".into());
        out.extend(main_words.iter().enumerate().map(|(i, w)| if i == 0 && !is_proper(&t[main[0]]) { lower_first(w) } else { w.clone() }));
    }
    if let Some(p) = end_punct {
        out.push(t[p].form.clone());
    }
    Some(out)
}

/// Same action, different modality. Negated clauses are left alone.
fn reframes(s: &Sentence, kept: &[usize]) -> Vec<Vec<String>> {
    let t = &s.tokens;
    let mut out = Vec::new();
    for &v in kept {
        if t[v].upos != "VERB" {
            continue;
        }
        let kids: Vec<usize> = s.children(v).collect();
        if kids.iter().any(|&k| matches!(t[k].lemma.to_lowercase().as_str(), "not" | "never")) {
            continue;
        }
        // Case A: modal auxiliary on the verb ("should run").
        let modal = kids.iter().copied().find(|&k| t[k].deprel == "aux" && MODALS.contains(&t[k].lemma.to_lowercase().as_str()));
        // Case B: want/need/have + to + verb ("want to run").
        let catena = (t[v].deprel == "xcomp")
            .then_some(t[v].head)
            .flatten()
            .filter(|&h| matches!(t[h].lemma.as_str(), "want" | "need" | "have") && !s.children(h).any(|k| matches!(t[k].deprel.as_str(), "aux") || t[k].lemma == "not"))
            .and_then(|h| kids.iter().copied().find(|&k| t[k].deprel == "mark" && t[k].form.eq_ignore_ascii_case("to")).map(|to| (h, to)));
        let (remove, current, subject_of): (Vec<usize>, String, usize) = match (modal, catena) {
            (Some(m), _) => (vec![m], t[m].lemma.to_lowercase(), v),
            (None, Some((h, to))) => (vec![h, to], t[h].lemma.clone(), h),
            _ => continue,
        };
        if !remove.iter().all(|r| kept.contains(r)) || t[v].xpos != "VB" {
            continue;
        }
        let third = s.children(subject_of).find(|&k| t[k].deprel == "nsubj").is_some_and(|k| {
            matches!(t[k].form.to_lowercase().as_str(), "he" | "she" | "it") || matches!(t[k].xpos.as_str(), "NN" | "NNP")
        });
        let past = remove.first().is_some_and(|&r| matches!(t[r].xpos.as_str(), "VBD"));
        let options: Vec<(&str, Vec<&str>)> = vec![
            ("should", vec!["should"]),
            ("want", vec![if past { "wanted" } else if third { "wants" } else { "want" }, "to"]),
            ("need", vec![if past { "needed" } else if third { "needs" } else { "need" }, "to"]),
            ("will", vec!["will"]),
            ("can", vec!["can"]),
            ("can't", vec!["ca", "n't"]),
        ];
        for (name, phrase) in options {
            if name == current || (current == "have" && name == "need") {
                continue;
            }
            let first = remove[0];
            let mut tokens = Vec::new();
            for &i in kept {
                if i == first {
                    let cap = t[i].form.chars().next().is_some_and(char::is_uppercase);
                    tokens.extend(phrase.iter().enumerate().map(|(j, w)| if j == 0 && cap { capitalize(w) } else { w.to_string() }));
                } else if !remove.contains(&i) {
                    tokens.push(t[i].form.clone());
                }
            }
            out.push(tokens);
        }
    }
    out
}

fn words(s: &Sentence, kept: &[usize]) -> Vec<String> {
    let t = &s.tokens;
    let mut out: Vec<String> = kept.iter().map(|&i| t[i].form.clone()).collect();
    while out.first().is_some_and(|w| matches!(w.as_str(), "," | ";")) {
        out.remove(0);
    }
    let end = t.last().map(|x| x.form.as_str()).filter(|f| matches!(*f, "." | "!" | "?"));
    if let (Some(end), Some(last)) = (end, out.last()) {
        if !matches!(last.as_str(), "." | "!" | "?") {
            while out.last().is_some_and(|w| matches!(w.as_str(), "," | ";")) {
                out.pop();
            }
            out.push(end.to_string());
        }
    }
    if let Some(first) = out.first_mut() {
        *first = capitalize(first);
    }
    out
}

/// "a" / "an" before the word at `pos` (or the word after an adjective there).
fn fix_article(tokens: &mut [String], pos: usize) {
    let Some(prev) = pos.checked_sub(1) else { return };
    let article = tokens[prev].to_lowercase();
    if article != "a" && article != "an" {
        return;
    }
    let next = tokens[pos].to_lowercase();
    let an = starts_with_vowel_sound(&next);
    let fixed = if an { "an" } else { "a" };
    tokens[prev] = if tokens[prev].starts_with('A') { capitalize(fixed) } else { fixed.to_string() };
}

fn starts_with_vowel_sound(w: &str) -> bool {
    const VOWEL_EXCEPTIONS: &[&str] = &["uni", "use", "usu", "uti", "ure", "eu", "one", "once", "ewe"];
    const SILENT_H: &[&str] = &["hour", "honest", "honor", "honour", "heir"];
    if SILENT_H.iter().any(|p| w.starts_with(p)) {
        return true;
    }
    if VOWEL_EXCEPTIONS.iter().any(|p| w.starts_with(p)) {
        return false;
    }
    w.starts_with(['a', 'e', 'i', 'o', 'u'])
}

fn capitalize(w: &str) -> String {
    let mut c = w.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

fn lower_first(w: &str) -> String {
    if w == "I" || w.starts_with("I'") {
        return w.to_string();
    }
    let mut c = w.chars();
    c.next().map(|f| f.to_lowercase().chain(c).collect()).unwrap_or_default()
}

fn is_proper(t: &crate::engine::Token) -> bool {
    t.upos == "PROPN" || t.form == "I"
}

fn join_sentences(sents: &[Vec<String>]) -> String {
    join_with_offsets(sents).0
}

/// Detokenized text plus each token's byte range in it.
fn join_with_offsets(sents: &[Vec<String>]) -> (String, Vec<(usize, usize)>) {
    let mut text = String::new();
    let mut offsets = Vec::new();
    for sent in sents {
        if !text.is_empty() {
            text.push(' ');
        }
        let start = text.len();
        let piece = crate::text::detokenize(sent);
        // Recover per-token ranges by scanning the detokenized piece.
        let lower = piece.to_lowercase();
        let mut cursor = 0;
        for tok in sent {
            let needle = tok.to_lowercase();
            match lower[cursor..].find(&needle) {
                Some(at) => {
                    let b = cursor + at;
                    offsets.push((start + b, start + b + needle.len()));
                    cursor = b + needle.len();
                }
                None => offsets.push((start + cursor, start + cursor)),
            }
        }
        text.push_str(&piece);
    }
    (text, offsets)
}

/// Ranges of tokens in `new` that don't line up with `old` (longest common subsequence).
fn changed_ranges(old: &[String], new: &[String], offsets: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let (n, m) = (old.len(), new.len());
    let mut dp = vec![vec![0u16; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i][j] = if old[i].eq_ignore_ascii_case(&new[j]) { dp[i + 1][j + 1] + 1 } else { dp[i + 1][j].max(dp[i][j + 1]) };
        }
    }
    let (mut i, mut j) = (0, 0);
    let mut matched = vec![false; m];
    while i < n && j < m {
        if old[i].eq_ignore_ascii_case(&new[j]) {
            matched[j] = true;
            i += 1;
            j += 1;
        } else if dp[i + 1][j] >= dp[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for (k, &(a, b)) in offsets.iter().enumerate() {
        if matched[k] || a == b || matches!(new[k].as_str(), "." | "," | "!" | "?") {
            continue;
        }
        match ranges.last_mut() {
            Some(last) if a <= last.1 + 1 => last.1 = b,
            _ => ranges.push((a, b)),
        }
    }
    ranges
}

/// Default ordering: best fluency first, then each next pick trades fluency
/// against similarity to what's already been picked. (Product-tunable.)
fn diversify(mut pool: Vec<Variant>, limit: usize) -> Vec<Variant> {
    let words = |v: &Variant| -> HashSet<String> { v.text.to_lowercase().split_whitespace().map(String::from).collect() };
    let sets: Vec<HashSet<String>> = pool.iter().map(words).collect();
    let mut order: Vec<usize> = Vec::new();
    let mut left: Vec<usize> = (0..pool.len()).collect();
    while !left.is_empty() && order.len() < limit {
        let pick = left
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let score = |x: usize| {
                    let sim = order.iter().map(|&o| jaccard(&sets[x], &sets[o])).fold(0.0, f64::max);
                    pool[x].fluency_delta - 1.5 * sim
                };
                score(a).total_cmp(&score(b))
            })
            .unwrap();
        order.push(pick);
        left.retain(|&x| x != pick);
    }
    let mut taken: Vec<Option<Variant>> = pool.drain(..).map(Some).collect();
    order.into_iter().filter_map(|i| taken[i].take()).collect()
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let inter = a.intersection(b).count() as f64;
    let union = (a.len() + b.len()) as f64 - inter;
    if union == 0.0 { 1.0 } else { inter / union }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split(' ').map(String::from).collect()
    }

    #[test]
    fn joins_two_sentences_with_a_connective() {
        assert_eq!(join_with(&toks("I stayed home ."), &toks("It rained ."), "because", false), toks("I stayed home because it rained ."));
        assert_eq!(join_with(&toks("I was tired ."), &toks("I went to bed ."), "so", false), toks("I was tired , so I went to bed ."));
        assert_eq!(join_with(&toks("I tried ."), &toks("Maria left !"), "but", true), toks("I tried , but Maria left !"));
    }

    #[test]
    fn only_joins_plain_statements() {
        assert!(can_join(&toks("I stayed home ."), &toks("It rained .")));
        assert!(!can_join(&toks("Why did I stay ?"), &toks("It rained .")));
        assert!(!can_join(&toks("I stayed ."), &toks("Because it rained .")));
        assert!(!can_join(&toks("I stayed ."), &toks("But it rained .")));
    }

    #[test]
    fn keeps_only_swaps_that_fit_nearly_as_well_as_the_original() {
        let scored = vec![("exhausted".to_string(), -3.0), ("gone".to_string(), -9.0), ("weary".to_string(), -4.5), ("spent".to_string(), -4.0), ("beat".to_string(), -4.2)];
        assert_eq!(keep_fitting(-2.5, scored, 2.5, 3), vec!["exhausted", "spent", "beat"]);
    }

    #[test]
    fn keeps_nothing_when_nothing_fits() {
        assert!(keep_fitting(-1.0, vec![("gone".to_string(), -8.0)], 2.5, 3).is_empty());
    }

    #[test]
    fn article_follows_the_next_sound() {
        let mut t: Vec<String> = ["I", "need", "a", "interruption"].map(String::from).to_vec();
        fix_article(&mut t, 3);
        assert_eq!(t[2], "an");
        let mut t: Vec<String> = ["an", "university"].map(String::from).to_vec();
        fix_article(&mut t, 1);
        assert_eq!(t[0], "a");
        let mut t: Vec<String> = ["a", "honest", "answer"].map(String::from).to_vec();
        fix_article(&mut t, 1);
        assert_eq!(t[0], "an");
    }
}
