//! Structure only, no vocabulary: each input line (a text) becomes its
//! sentences' dependency relations, with punctuation kept as the mark itself
//! so a comma and a period differ. One JSON line per input line: a list of
//! sentences, each a list of relations.
//!
//!   cargo run --release --example structure_parse -- <in.txt> <out.jsonl>
use scribe::{Engine, text};
use serde_json::json;
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, input, output] = &args[..] else {
        return Err("usage: structure_parse <in.txt> <out.jsonl>".into());
    };
    let e = Engine::load(Path::new("models"))?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(output)?);
    for line in std::fs::read_to_string(input)?.lines() {
        let sentences: Vec<Vec<String>> = text::sentences(line)
            .iter()
            .map(|s| {
                let toks = text::tokenize(s);
                let refs: Vec<&str> = toks.iter().map(String::as_str).collect();
                let tags = e.upos.tag(&refs, Some(&e.lexicon));
                let arcs = e.parser.parse(&refs, &tags);
                arcs.iter()
                    .zip(&toks)
                    .map(|(a, t)| {
                        if t.chars().any(char::is_alphanumeric) {
                            a.label.split(':').next().unwrap_or("").to_string()
                        } else {
                            t.clone()
                        }
                    })
                    .collect()
            })
            .collect();
        writeln!(out, "{}", json!(sentences))?;
    }
    Ok(())
}
