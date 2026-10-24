//! Open English WordNet synsets as JSON lines (for scripts that generalise
//! over meanings): id ("s<index>", as other examples write synsets), part of
//! speech, lexicographer category, members, hypernyms and similar adjectives.
//!
//!   cargo run --release --example wordnet_export -- <out.jsonl>
use scribe::lexicon::Lexicon;
use serde_json::json;
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args().nth(1).ok_or("usage: wordnet_export <out.jsonl>")?;
    let lex = Lexicon::load(Path::new("models/lexicon.bin"))?;
    let mut w = std::io::BufWriter::new(std::fs::File::create(out)?);
    let id = |i: &u32| format!("s{i}");
    for (i, s) in lex.synsets.iter().enumerate() {
        writeln!(w, "{}", json!({"id": format!("s{i}"), "pos": format!("{:?}", s.pos), "category": lex.category_name(i as u32),
            "members": s.members, "hypernyms": s.hypernyms.iter().map(id).collect::<Vec<_>>(), "similar": s.similar.iter().map(id).collect::<Vec<_>>()}))?;
    }
    eprintln!("{} synsets", lex.synsets.len());
    Ok(())
}
