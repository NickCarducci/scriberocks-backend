//! Synonym sets with shared usage samples (see src/usage/synonyms.rs), built
//! from WordNet synsets and the usage corpus, scored by the usage model.
//!
//!   cargo run --release --example synonym_sets [synsets] [per-member] [margin] [skip]
//!
//! Writes data/synonyms/shared.jsonl: one set per line with its shared
//! sentences, and prints coverage plus a few examples.
use scribe::usage::synonyms;
use scribe::{text, Engine};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: f64| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (n_synsets, per_member, margin, skip) = (arg(1, 300.0) as usize, arg(2, 10.0) as usize, arg(3, 2.0) as f32, arg(4, 0.0) as usize);
    let e = Engine::load(Path::new("models")).unwrap();
    let usage = e.usage.as_ref().unwrap();

    // Usage corpus: Wiktionary + WordNet examples + EWT, lowercased tokens.
    let tokenize = |s: &str| -> Vec<String> { text::tokenize(s).into_iter().map(|w| w.to_lowercase()).collect() };
    let mut corpus: Vec<Vec<String>> = std::fs::read_to_string("data/wiktionary/examples.txt").unwrap().lines().map(tokenize).collect();
    corpus.extend(e.lexicon.synsets.iter().flat_map(|s| s.examples.iter().map(|x| tokenize(x))));
    let ewt = scribe::conllu::parse(&std::fs::read_to_string("data/en_ewt-ud-train.conllu").unwrap());
    corpus.extend(ewt.iter().map(|s| s.iter().map(|t| t.form.to_lowercase()).collect()));

    // Word frequency, to start with synsets of common words.
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for s in &corpus { for w in s { *freq.entry(w.as_str()).or_default() += 1 } }
    let mut candidates: Vec<(u32, Vec<String>, usize)> = e
        .lexicon
        .synsets
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let members: Vec<String> = s.members.iter().map(|m| m.to_lowercase()).filter(|m| !m.contains(' ') && usage.vocab.contains(m)).collect();
            (members.len() >= 2).then(|| {
                let least = members.iter().map(|m| freq.get(m.as_str()).copied().unwrap_or(0)).min().unwrap_or(0);
                (i as u32, members, least)
            })
        })
        .collect();
    candidates.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    let picked: Vec<(u32, Vec<String>)> = candidates.iter().skip(skip).take(n_synsets).map(|(s, m, _)| (*s, m.clone())).collect();
    eprintln!("{} synsets with 2+ single-word members in the vocabulary; trying the {} whose rarest member is most common", candidates.len(), picked.len());

    let t = Instant::now();
    let sets = synonyms::build(&picked, &corpus, usage as &dyn scribe::engine::SlotScorer, per_member, margin);
    let with = sets.iter().filter(|s| !s.samples.is_empty()).count();
    let samples: usize = sets.iter().map(|s| s.samples.len()).sum();
    let tried: usize = picked.iter().map(|(_, m)| m.len() * per_member).sum();
    eprintln!(
        "{:.1?}: {with}/{} sets have shared samples; {samples} shared samples from ≤{tried} tried sentences",
        t.elapsed(), sets.len()
    );

    std::fs::create_dir_all("data/synonyms").unwrap();
    let mut out = std::io::BufWriter::new(std::fs::File::create("data/synonyms/shared.jsonl").unwrap());
    for s in &sets {
        let samples: Vec<serde_json::Value> = s.samples.iter().map(|x| serde_json::json!({ "text": corpus[x.sentence].join(" "), "slot": x.slot, "fits": x.fits })).collect();
        writeln!(out, "{}", serde_json::json!({ "synset": e.lexicon.synsets[s.synset as usize].id, "definition": e.lexicon.synsets[s.synset as usize].definition, "members": s.members, "samples": samples })).unwrap();
    }
    for s in sets.iter().filter(|s| s.samples.len() >= 2).take(6) {
        println!("\n{} — {}", s.members.join(", "), e.lexicon.synsets[s.synset as usize].definition);
        for x in s.samples.iter().take(3) {
            let mut words = corpus[x.sentence].clone();
            words[x.slot] = format!("[{}]", words[x.slot]);
            println!("   {}   fits: {}", words.join(" "), x.fits.join(", "));
        }
    }
}
