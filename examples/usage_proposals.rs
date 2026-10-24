//! A usage model's own proposals for each substitution slot, as rows for
//! substitution_records (a FidelityAI source built by the usage model; the
//! human-judged substitutes are the candidates). For each sentence of the
//! input rows (prolex_substitutions.py format), the model's top-k words for
//! the target's slot, target visible, single words other than the target
//! that the dictionary lists with the target's part of speech (as tagged in
//! its sentence).
//!
//!   cargo run --release --example usage_proposals -- <subs.tsv> <out.tsv> [model = models/casual-v2] [k = 10]
//!
//! ProLex sentences are ETS-licensed: rulebooks built here stay local
//! evaluation inputs, discarded after scoring, never shipped.
use scribe::bert::Bert;
use scribe::{Engine, Pos};
use std::collections::HashSet;
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (inp, outp) = (args.get(1).ok_or("usage: usage_proposals <subs.tsv> <out.tsv> [model] [k]")?, args.get(2).ok_or("out")?);
    let model = args.get(3).map(String::as_str).unwrap_or("models/casual-v2");
    let k: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(10);
    let bert = Bert::load(Path::new(model), scribe::usage::model::Device::Cpu)?;
    let e = Engine::load(Path::new("models"))?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(outp)?);
    let (mut seen, mut n) = (HashSet::new(), 0);
    for line in std::fs::read_to_string(inp)?.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let [_, tid, _, prefix, target, _, suffix] = c[..] else { continue };
        if !seen.insert(tid.to_string()) {
            continue;
        }
        let at = scribe::text::tokenize(prefix).len();
        let words: Vec<String> = scribe::text::tokenize(&format!("{prefix} {target} {suffix}")).iter().map(|w| w.to_lowercase()).collect();
        if words.get(at).map(String::as_str) != Some(target.to_lowercase().as_str()) {
            continue;
        }
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let pos = Pos::from_upos(e.upos.tag(&refs, Some(&e.lexicon))[at]);
        let listed = |w: &str| match pos {
            Some(p) => e.lexicon.lemmatize(w, Some(p)).iter().any(|l| !e.lexicon.senses(l, p).is_empty()),
            None => true,
        };
        let fills = bert.predict(&words, at, 4 * k, true)?;
        let target = target.to_lowercase();
        for (w, _) in fills.into_iter().filter(|(w, _)| *w != target && w.chars().all(char::is_alphabetic) && listed(w)).take(k) {
            writeln!(out, "usage\t{tid}\t1\t{prefix}\t{target}\t{w}\t{suffix}")?;
            n += 1;
        }
    }
    eprintln!("{} sentences, {n} proposals from {model}", seen.len());
    Ok(())
}
