//! Closed factors of each clause, for searching which combinations of English
//! a rulebook should hold: fine tags and function-word forms only, read along
//! parser links within a sentence. Each input line becomes one JSON line: a
//! list of clauses, each a map of factor to value.
//!   S subject tag, D subject's determiner, A auxiliary or copula, V verb tag,
//!   O object tag, Q object's determiner, N negated, C the clause's relation,
//!   M its subordinator, P first oblique's preposition
//!
//!   cargo run --release --example clause_factors -- <in.txt> <out.jsonl>
use scribe::{Engine, text};
use serde_json::{Map, Value, json};
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, input, output] = &args[..] else {
        return Err("usage: clause_factors <in.txt> <out.jsonl>".into());
    };
    let e = Engine::load(Path::new("models"))?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(output)?);
    for line in std::fs::read_to_string(input)?.lines() {
        let mut clauses = Vec::<Value>::new();
        for s in text::sentences(line) {
            let toks = text::tokenize(&s);
            let refs: Vec<&str> = toks.iter().map(String::as_str).collect();
            let up = e.upos.tag(&refs, Some(&e.lexicon));
            let xp = e.xpos.tag(&refs, Some(&e.lexicon));
            let arcs = e.parser.parse(&refs, &up);
            let rel = |j: usize| arcs[j].label.split(':').next().unwrap_or("");
            let kids = |h: usize| -> Vec<usize> { (0..arcs.len()).filter(|&j| arcs[j].head == h + 1).collect() };
            let low = |j: usize| toks[j].to_lowercase();
            let child = |h: usize, rs: &[&str]| kids(h).into_iter().find(|&k| rs.contains(&rel(k)));
            let form = |j: Option<usize>| j.map_or("-".to_string(), low);
            let tag = |j: Option<usize>| j.map_or("-".to_string(), |j| xp[j].to_string());
            for h in 0..toks.len() {
                if !(up[h] == "VERB" || child(h, &["cop"]).is_some()) {
                    continue;
                }
                let subj = child(h, &["nsubj"]);
                let obj = child(h, &["obj"]);
                let obl = child(h, &["obl"]);
                let neg = kids(h).into_iter().any(|k| matches!(low(k).as_str(), "not" | "n't" | "never"));
                let mut c = Map::new();
                c.insert("S".into(), json!(tag(subj)));
                c.insert("D".into(), json!(form(subj.and_then(|j| child(j, &["det"])))));
                c.insert("A".into(), json!(form(child(h, &["aux", "cop"]))));
                c.insert("V".into(), json!(xp[h]));
                c.insert("O".into(), json!(tag(obj)));
                c.insert("Q".into(), json!(form(obj.and_then(|j| child(j, &["det"])))));
                c.insert("N".into(), json!(if neg { "y" } else { "n" }));
                c.insert("C".into(), json!(rel(h)));
                c.insert("M".into(), json!(form(child(h, &["mark"]))));
                c.insert("P".into(), json!(form(obl.and_then(|j| child(j, &["case"])))));
                clauses.push(Value::Object(c));
            }
        }
        writeln!(out, "{}", Value::Array(clauses))?;
    }
    Ok(())
}
