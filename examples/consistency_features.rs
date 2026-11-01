//! A text's sentences as closed writing choices, for checking whether a piece agrees
//! with itself. One JSON line per input line: a list of sentences, each a map of
//! T tense, P person, C contraction, V voice, O opener, H hedge, N nominal density,
//! L length band, D digits.
//!
//!   cargo run --release --example consistency_features -- <in.txt> <out.jsonl>
use scribe::{Engine, text};
use serde_json::{Map, Value, json};
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let e = Engine::load(Path::new("models"))?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[2])?);
    for line in std::fs::read_to_string(&a[1])?.lines() {
        let mut doc = Vec::<Value>::new();
        for s in text::sentences(line) {
            let toks = text::tokenize(&s);
            if toks.is_empty() {
                continue;
            }
            let refs: Vec<&str> = toks.iter().map(String::as_str).collect();
            let up = e.upos.tag(&refs, Some(&e.lexicon));
            let xp = e.xpos.tag(&refs, Some(&e.lexicon));
            let arcs = e.parser.parse(&refs, &up);
            let low: Vec<String> = toks.iter().map(|w| w.to_lowercase()).collect();
            let root = arcs.iter().position(|x| x.head == 0).unwrap_or(0);
            let kids: Vec<usize> = (0..arcs.len()).filter(|&j| arcs[j].head == root + 1).collect();
            let aux: Vec<&str> = kids.iter().filter(|&&k| arcs[k].label.starts_with("aux") || arcs[k].label == "cop").map(|&k| low[k].as_str()).collect();
            let tense = if aux.iter().any(|w| matches!(*w, "will" | "'ll" | "shall")) {
                "future"
            } else if aux.iter().any(|w| matches!(*w, "was" | "were" | "had" | "did" | "would" | "could" | "might" | "should")) || xp[root] == "VBD" {
                "past"
            } else if !aux.is_empty() || matches!(xp[root], "VBP" | "VBZ") {
                "present"
            } else {
                "none"
            };
            let subj = kids.iter().find(|&&k| arcs[k].label.starts_with("nsubj")).map(|&k| k);
            let person = match subj.map(|k| low[k].as_str()) {
                Some("i") | Some("we") => "first",
                Some("you") => "second",
                Some("one") => "one",
                Some(_) => "third",
                None => "none",
            };
            let contraction = low.iter().any(|w| matches!(w.as_str(), "n't" | "'re" | "'ll" | "'ve" | "'d" | "'m"));
            // A form of "be" on a past-participle root: the parser's labels don't mark passive.
            let passive = xp[root] == "VBN" && aux.iter().any(|w| matches!(*w, "is" | "are" | "was" | "were" | "be" | "been" | "being" | "am"));
            let hedge = low.iter().any(|w| matches!(w.as_str(), "may" | "might" | "could" | "perhaps" | "maybe" | "possibly" | "probably" | "seems"));
            let words: Vec<&String> = low.iter().filter(|w| w.chars().any(char::is_alphanumeric)).collect();
            let nominal = words.iter().filter(|w| ["tion", "ment", "ness", "ity", "ance", "ence"].iter().any(|x| w.len() > x.len() + 2 && w.ends_with(x))).count();
            let digits = low.iter().any(|w| w.chars().any(|c| c.is_ascii_digit()));
            let mut c = Map::new();
            c.insert("T".into(), json!(tense));
            c.insert("P".into(), json!(person));
            c.insert("C".into(), json!(if contraction { "y" } else { "n" }));
            c.insert("V".into(), json!(if passive { "passive" } else { "active" }));
            c.insert("O".into(), json!(up[0]));
            c.insert("H".into(), json!(if hedge { "y" } else { "n" }));
            c.insert("N".into(), json!(nominal.min(3).to_string()));
            c.insert("L".into(), json!((words.len() / 8).min(5).to_string()));
            c.insert("D".into(), json!(if digits { "y" } else { "n" }));
            doc.push(Value::Object(c));
        }
        writeln!(out, "{}", Value::Array(doc))?;
    }
    Ok(())
}
