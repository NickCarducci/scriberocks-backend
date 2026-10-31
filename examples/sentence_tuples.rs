//! Each text as its sequence of parser-linked word tuples, in sentence order:
//! (governor lemma, relation, dependent lemma), punctuation left out. One JSON line
//! per input line.
//!
//!   cargo run --release --example sentence_tuples -- <in.txt> <out.jsonl>
use scribe::lexicon::Pos;
use scribe::{Engine, text};
use serde_json::json;
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let e = Engine::load(Path::new("models"))?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[2])?);
    for line in std::fs::read_to_string(&a[1])?.lines() {
        let mut tuples = Vec::new();
        for s in text::sentences(line) {
            let toks = text::tokenize(&s);
            let refs: Vec<&str> = toks.iter().map(String::as_str).collect();
            let tags = e.upos.tag(&refs, Some(&e.lexicon));
            let arcs = e.parser.parse(&refs, &tags);
            let lemma = |i: usize| match Pos::from_upos(tags[i]) {
                Some(p) if tags[i] != "PROPN" => e.lexicon.lemmatize(&toks[i], Some(p)).into_iter().next(),
                _ => None,
            }
            .unwrap_or_else(|| toks[i].to_lowercase());
            for j in 0..toks.len() {
                if !toks[j].chars().any(char::is_alphanumeric) {
                    continue;
                }
                let g = arcs[j].head.checked_sub(1).map_or("<root>".to_string(), &lemma);
                tuples.push(json!([g, arcs[j].label.split(':').next().unwrap_or(""), lemma(j)]));
            }
        }
        writeln!(out, "{}", json!(tuples))?;
    }
    Ok(())
}
