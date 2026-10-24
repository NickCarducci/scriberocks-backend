//! Reference tags and arcs for checking FidelityAI's JavaScript port of the
//! tagger and parser (scribe_parse.js) against Rust: for each UD English EWT
//! test sentence (gold tokens), the pruned models' UPOS tags and arcs, one
//! JSON line each, plus the dictionary forms (lemmatize) of every word.
//!
//!   cargo run --release --example parse_parity_dump <out.jsonl> [prune = 1.0]
use scribe::{conllu, Engine};
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args().nth(1).ok_or("usage: parse_parity_dump <out.jsonl> [prune]")?;
    let prune: f32 = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let mut e = Engine::load(Path::new("models"))?;
    e.upos.prune(prune);
    e.parser.prune(prune);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out)?);
    let mut n = 0;
    for s in conllu::parse(&std::fs::read_to_string("data/en_ewt-ud-test.conllu")?) {
        let words: Vec<&str> = s.iter().map(|t| t.form.as_str()).collect();
        let tags = e.upos.tag(&words, Some(&e.lexicon));
        let arcs = e.parser.parse(&words, &tags);
        let lemmas: Vec<Vec<String>> = words.iter().map(|w| e.lexicon.lemmatize(w, None)).collect();
        let line = serde_json::json!({
            "words": words, "tags": tags,
            "heads": arcs.iter().map(|a| a.head).collect::<Vec<_>>(),
            "labels": arcs.iter().map(|a| a.label.clone()).collect::<Vec<_>>(),
            "lemmas": lemmas,
        });
        writeln!(f, "{line}")?;
        n += 1;
    }
    println!("{out}: {n} sentences");
    Ok(())
}
