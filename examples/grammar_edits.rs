//! Editors' grammar corrections from Simple English Wikipedia history, for
//! testing what FidelityAI catches. Real edits only (article id, before,
//! after; extract_simplewiki_edits.py), sorted into:
//!   split       a run-on sentence split in two: the same words, one more
//!               sentence, at most one joining word (and, but, so, which,
//!               because, while) dropped
//!   inflection  one word changed form, same lemma (is → are, go → went):
//!               agreement, tense, number
//!   function    one function word (a, the, in, of, is, ...) replaced,
//!               inserted or deleted
//! Each side is written with its tokens, fine part-of-speech tags (Penn Treebank,
//! one sequence per sentence) and per-sentence shape (words, commas,
//! joining words, verbs). Also writes a rulebook of stable sentences.
//!
//!   cargo run --release --example grammar_edits -- <out dir> [rulebook = 5000]
use scribe::lexicon::Pos;
use scribe::text;
use scribe::Engine;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

const JOINERS: [&str; 6] = ["and", "but", "so", "which", "because", "while"];
const FUNCTION: [&str; 30] = [
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "with", "by", "from", "into", "about", "as", "is", "are", "was", "were",
    "be", "been", "has", "have", "had", "it", "its", "this", "that", "these", "those",
];

fn words(s: &str) -> Vec<String> {
    text::tokenize(s).into_iter().map(|w| w.to_lowercase()).filter(|w| w.chars().any(char::is_alphanumeric)).collect()
}

fn describe(e: &Engine, s: &str) -> Value {
    let sentences: Vec<Value> = text::sentences(s)
        .iter()
        .map(|sent| {
            let toks = text::tokenize(sent);
            let refs: Vec<&str> = toks.iter().map(String::as_str).collect();
            let xpos = e.xpos.tag(&refs, Some(&e.lexicon));
            let upos = e.upos.tag(&refs, Some(&e.lexicon));
            let lower: Vec<String> = toks.iter().map(|w| w.to_lowercase()).collect();
            json!({
                "tokens": toks,
                "xpos": xpos,
                "shape": {
                    "words": lower.iter().filter(|w| w.chars().any(char::is_alphanumeric)).count(),
                    "commas": lower.iter().filter(|w| *w == ",").count(),
                    "joiners": lower.iter().filter(|w| JOINERS.contains(&w.as_str())).count(),
                    "verbs": upos.iter().filter(|t| **t == "VERB" || **t == "AUX").count(),
                }
            })
        })
        .collect();
    json!({"text": s, "sentences": sentences})
}

fn lemmas(e: &Engine, w: &str) -> HashSet<String> {
    [Pos::Noun, Pos::Verb, Pos::Adj, Pos::Adv].iter().flat_map(|&p| e.lexicon.lemmatize(w, Some(p))).collect()
}

fn kind(e: &Engine, before: &str, after: &str) -> Option<&'static str> {
    let (b, a) = (words(before), words(after));
    if let Some(k) = split(before, after) {
        return Some(k);
    }
    if text::sentences(after).len() != text::sentences(before).len() {
        return None;
    }
    if a.len() == b.len() {
        let diff: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
        if diff.len() != 1 {
            return None;
        }
        let (x, y) = (&b[diff[0]], &a[diff[0]]);
        if FUNCTION.contains(&x.as_str()) && FUNCTION.contains(&y.as_str()) {
            return Some("function");
        }
        let (lx, ly) = (lemmas(e, x), lemmas(e, y));
        return (x.chars().all(char::is_alphabetic) && y.chars().all(char::is_alphabetic) && !lx.is_disjoint(&ly)).then_some("inflection");
    }
    // One function word inserted or deleted.
    let (long, short) = if a.len() > b.len() { (&a, &b) } else { (&b, &a) };
    if long.len() == short.len() + 1 {
        let i = (0..short.len()).find(|&i| long[i] != short[i]).unwrap_or(short.len());
        if long[..i] == short[..i] && long[i + 1..] == short[i..] && FUNCTION.contains(&long[i].as_str()) {
            return Some("function");
        }
    }
    None
}

/// A run-on split: the edit inserts "." after an ordinary word (four or more
/// letters, so not an abbreviation's period) before a word (capitalised or not), and
/// may drop one comma or joining word there; nothing else changes. Both
/// sentences have at least five words.
fn split(before: &str, after: &str) -> Option<&'static str> {
    let (b, a) = (text::tokenize(before), text::tokenize(after));
    let i = (0..b.len().min(a.len())).find(|&i| b[i] != a[i])?;
    if a.get(i).map(String::as_str) != Some(".") || i == 0 {
        return None;
    }
    let prev = &a[i - 1];
    if prev.chars().count() < 4 || !prev.chars().all(char::is_alphabetic) {
        return None;
    }
    let next = a.get(i + 1)?;
    if !next.chars().next().is_some_and(char::is_alphabetic) {
        return None;
    }
    // Before, from i: optional "," / joiner, then the same word (any case), then the same rest.
    let mut j = i;
    if b.get(j).is_some_and(|w| w == "," || JOINERS.contains(&w.to_lowercase().as_str())) {
        j += 1;
    }
    if b.get(j).is_some_and(|w| w == ",") {
        j += 1;
    }
    let same_next = b.get(j).is_some_and(|w| w.to_lowercase() == next.to_lowercase());
    let same_rest = b.get(j + 1..) == a.get(i + 2..);
    let long = |ws: &[String]| ws.iter().filter(|w| w.chars().any(char::is_alphanumeric)).count() >= 5;
    (same_next && same_rest && long(&a[..i]) && long(&a[i + 1..])).then_some("split")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).ok_or("usage: grammar_edits <out dir> [rulebook]")?;
    let n_rulebook: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(5000);
    let e = Engine::load(Path::new("models"))?;
    std::fs::create_dir_all(out)?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(format!("{out}/edits.jsonl"))?);
    let mut counts = std::collections::BTreeMap::new();
    for file in ["train", "dev", "test"] {
        for line in std::fs::read_to_string(format!("data/simplewiki-history/{file}.tsv"))?.lines() {
            let c: Vec<&str> = line.split('\t').collect();
            let [id, before, after] = c[..] else { continue };
            if let Some(k) = kind(&e, before, after) {
                *counts.entry(k).or_insert(0) += 1;
                writeln!(w, "{}", json!({"id": format!("{file}:{}", n_edits(&counts)), "article": id, "kind": k,
                    "before": describe(&e, before), "after": describe(&e, after)}))?;
            }
        }
    }
    eprintln!("edits: {counts:?}");
    let mut r = std::io::BufWriter::new(std::fs::File::create(format!("{out}/rulebook.jsonl"))?);
    let stable: Vec<String> = std::fs::read_to_string("data/simplewiki-history/stable_train.tsv")?
        .lines()
        .filter_map(|l| l.split('\t').nth(3).map(String::from))
        .collect();
    let step = (stable.len() / n_rulebook).max(1);
    for s in stable.iter().step_by(step).take(n_rulebook) {
        writeln!(r, "{}", describe(&e, s))?;
    }
    Ok(())
}

fn n_edits(counts: &std::collections::BTreeMap<&str, i32>) -> i32 {
    counts.values().sum()
}
