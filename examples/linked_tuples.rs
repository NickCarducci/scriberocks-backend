//! Tuples of words the parser links, at any distance, as closed features only:
//! fine part-of-speech tags and function-word forms, never open vocabulary.
//! Each input line (a text) becomes one JSON line: its tokens, and tuples with
//! the token positions they read (positions count across the text's sentences).
//!   pairs   subject-finite verb, determiner-noun, auxiliary-verb (2 values)
//!   clause  subject, auxiliary or copula, verb, object (4 values)
//!   np      determiner, modifier kind, noun, the noun's relation (4 values)
//!
//!   cargo run --release --example linked_tuples -- <in.txt> <out.jsonl>
use scribe::{Engine, text};
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, input, output] = &args[..] else {
        return Err("usage: linked_tuples <in.txt> <out.jsonl>".into());
    };
    let e = Engine::load(Path::new("models"))?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(output)?);
    for line in std::fs::read_to_string(input)?.lines() {
        let (mut all, mut pairs, mut clause, mut np) = (Vec::new(), Vec::<Value>::new(), Vec::<Value>::new(), Vec::<Value>::new());
        for s in text::sentences(line) {
            let toks = text::tokenize(&s);
            let refs: Vec<&str> = toks.iter().map(String::as_str).collect();
            let up = e.upos.tag(&refs, Some(&e.lexicon));
            let xp = e.xpos.tag(&refs, Some(&e.lexicon));
            let arcs = e.parser.parse(&refs, &up);
            let base = all.len();
            let rel = |j: usize| arcs[j].label.split(':').next().unwrap_or("");
            let kids = |h: usize| -> Vec<usize> { (0..arcs.len()).filter(|&j| arcs[j].head == h + 1).collect() };
            let low = |j: usize| toks[j].to_lowercase();
            let child = |h: usize, rs: &[&str]| kids(h).into_iter().find(|&k| rs.contains(&rel(k)));
            for h in 0..toks.len() {
                let cop = child(h, &["cop"]);
                if up[h] == "VERB" || cop.is_some() {
                    let subj = child(h, &["nsubj"]);
                    let aux = child(h, &["aux"]).or(cop);
                    let obj = child(h, &["obj"]);
                    let finite = aux.unwrap_or(h);
                    if let Some(sj) = subj {
                        let f = if aux.is_some() { format!("F:{}", low(finite)) } else { format!("F:{}", xp[h]) };
                        pairs.push(json!({"v": [format!("S:{}", xp[sj]), f], "at": [base + sj, base + finite]}));
                    }
                    if let Some(a) = aux {
                        pairs.push(json!({"v": [format!("A:{}", low(a)), format!("V:{}", xp[h])], "at": [base + a, base + h]}));
                    }
                    let mut at = vec![base + h];
                    at.extend([subj, aux, obj].iter().flatten().map(|j| base + j));
                    clause.push(json!({"v": [
                        subj.map_or("S:-".to_string(), |j| format!("S:{}", xp[j])),
                        aux.map_or("A:-".to_string(), |j| format!("A:{}", low(j))),
                        format!("V:{}", xp[h]),
                        obj.map_or("O:-".to_string(), |j| format!("O:{}", xp[j])),
                    ], "at": at}));
                }
                if up[h] == "NOUN" || up[h] == "PROPN" {
                    let det = child(h, &["det"]);
                    if let Some(d) = det {
                        pairs.push(json!({"v": [format!("D:{}", low(d)), format!("N:{}", xp[h])], "at": [base + d, base + h]}));
                    }
                    let modifier = child(h, &["amod", "nummod", "compound", "nmod"]);
                    let mut at = vec![base + h];
                    at.extend([det, modifier].iter().flatten().map(|j| base + j));
                    np.push(json!({"v": [
                        det.map_or("D:-".to_string(), |j| format!("D:{}", low(j))),
                        modifier.map_or("M:-".to_string(), |j| format!("M:{}", rel(j))),
                        format!("N:{}", xp[h]),
                        format!("R:{}", rel(h)),
                    ], "at": at}));
                }
            }
            all.extend(toks);
        }
        writeln!(out, "{}", json!({"tokens": all, "pairs": pairs, "clause": clause, "np": np}))?;
    }
    Ok(())
}
