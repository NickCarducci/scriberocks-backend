//! Word pairings attested in usage text, for scoring substitutes: each
//! substitute's (governor, relation, word) triple and its (word, relation,
//! dependent) triples, counted in a corpus (Tatoeba, OANC, ...). A rejected
//! substitute is often a common word in a pairing the corpus never shows.
//!
//!   cargo run --release --example collocations -- <subs.tsv> <corpus.txt> <out prefix>
//!
//! subs.tsv: scripts/prolex_substitutions.py rows. Writes <out>.candidates.jsonl
//! (each substitute's triples with counts) and <out>.rulebook.tsv (corpus
//! triples sharing a candidate's governor and relation, or relation and
//! dependent, for a FidelityAI rulebook: real text only). Tagger and parser
//! only (no meaning model), on corpus and candidates alike.
use scribe::lexicon::Pos;
use scribe::Engine;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, Write};
use std::path::Path;

/// (lemmas, relations, heads) for one tokenized sentence.
fn light_parse(e: &Engine, words: &[String]) -> (Vec<String>, Vec<String>, Vec<Option<usize>>) {
    let refs: Vec<&str> = words.iter().map(String::as_str).collect();
    let tags = e.upos.tag(&refs, Some(&e.lexicon));
    let arcs = e.parser.parse(&refs, &tags);
    let lemmas = (0..words.len())
        .map(|i| match Pos::from_upos(tags[i]) {
            Some(p) if tags[i] != "PROPN" => e.lexicon.lemmatize(&words[i], Some(p)).into_iter().next(),
            _ => None,
        }
        .unwrap_or_else(|| words[i].to_lowercase()))
        .collect();
    let rels = arcs.iter().map(|a| a.label.split(':').next().unwrap_or("").to_string()).collect();
    let heads = arcs.iter().map(|a| a.head.checked_sub(1)).collect();
    (lemmas, rels, heads)
}

type Triple = (String, String, String); // governor, relation, dependent

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, subs, corpus, out] = &args[..] else { return Err("usage: collocations <subs.tsv> <corpus.txt> <out prefix>".into()) };
    let e = Engine::load(Path::new("models"))?;

    // Candidates: the substitute's triples (up: to its governor; down: to its dependents).
    let mut cands = Vec::new();
    for line in std::fs::read_to_string(subs)?.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let [split, tid, y, prefix, _target, sub, suffix] = c[..] else { continue };
        let at = scribe::text::tokenize(prefix).len();
        let words = scribe::text::tokenize(&format!("{prefix} {sub} {suffix}"));
        if words.get(at).map(|w| w.eq_ignore_ascii_case(sub)) != Some(true) {
            continue;
        }
        let (lemmas, rels, heads) = light_parse(&e, &words);
        let up: Option<Triple> = heads[at].map(|h| (lemmas[h].clone(), rels[at].clone(), lemmas[at].clone()));
        let down: Vec<Triple> = (0..words.len())
            .filter(|&j| heads[j] == Some(at) && !matches!(rels[j].as_str(), "punct" | "det" | "case" | "cc" | "aux" | "mark"))
            .map(|j| (lemmas[at].clone(), rels[j].clone(), lemmas[j].clone()))
            .collect();
        cands.push((split.to_string(), tid.to_string(), y.parse::<u8>()?, up, down));
    }
    let up_keys: HashSet<(String, String)> = cands.iter().filter_map(|c| c.3.as_ref()).map(|t| (t.0.clone(), t.1.clone())).collect();
    let down_keys: HashSet<(String, String)> = cands.iter().flat_map(|c| c.4.iter()).map(|t| (t.1.clone(), t.2.clone())).collect();
    eprintln!("{} candidates; {} governor keys, {} dependent keys", cands.len(), up_keys.len(), down_keys.len());

    // Corpus: count triples sharing a key; unigram lemma counts for PMI.
    let text = std::fs::read_to_string(corpus)?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let chunk = lines.len().div_ceil(threads);
    let partial: Vec<(HashMap<Triple, u32>, HashMap<String, u64>, u64)> = std::thread::scope(|s| {
        let handles: Vec<_> = lines
            .chunks(chunk)
            .map(|part| {
                let (e, up_keys, down_keys) = (&e, &up_keys, &down_keys);
                s.spawn(move || {
                    let (mut triples, mut unigrams, mut n) = (HashMap::new(), HashMap::new(), 0u64);
                    for line in part {
                        for sentence in scribe::text::sentences(line) {
                            let words = scribe::text::tokenize(&sentence);
                            if words.is_empty() || words.len() > 80 {
                                continue;
                            }
                            let (lemmas, rels, heads) = light_parse(e, &words);
                            for j in 0..words.len() {
                                *unigrams.entry(lemmas[j].clone()).or_insert(0u64) += 1;
                                n += 1;
                                let Some(h) = heads[j] else { continue };
                                let up = (lemmas[h].clone(), rels[j].clone());
                                let down = (rels[j].clone(), lemmas[j].clone());
                                if up_keys.contains(&up) || down_keys.contains(&down) {
                                    *triples.entry((up.0, up.1, down.1)).or_insert(0u32) += 1;
                                }
                            }
                        }
                    }
                    (triples, unigrams, n)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (mut triples, mut unigrams, mut tokens) = (HashMap::<Triple, u32>::new(), HashMap::<String, u64>::new(), 0u64);
    for (t, u, n) in partial {
        for (k, v) in t {
            *triples.entry(k).or_insert(0) += v;
        }
        for (k, v) in u {
            *unigrams.entry(k).or_insert(0) += v;
        }
        tokens += n;
    }
    let mut key_up: HashMap<(String, String), u32> = HashMap::new();
    let mut key_down: HashMap<(String, String), u32> = HashMap::new();
    for ((g, r, d), c) in &triples {
        *key_up.entry((g.clone(), r.clone())).or_insert(0) += c;
        *key_down.entry((r.clone(), d.clone())).or_insert(0) += c;
    }
    eprintln!("corpus: {} lines, {tokens} tokens, {} matching triples", lines.len(), triples.len());

    let count = |t: &Triple| triples.get(t).copied().unwrap_or(0);
    let mut w = std::io::BufWriter::new(std::fs::File::create(format!("{out}.candidates.jsonl"))?);
    for (split, tid, y, up, down) in &cands {
        let triple_json = |t: &Triple| {
            json!({"gov": t.0, "rel": t.1, "dep": t.2, "count": count(t),
                   "gov_rel": key_up.get(&(t.0.clone(), t.1.clone())).copied().unwrap_or(0),
                   "rel_dep": key_down.get(&(t.1.clone(), t.2.clone())).copied().unwrap_or(0),
                   "gov_n": unigrams.get(&t.0).copied().unwrap_or(0), "dep_n": unigrams.get(&t.2).copied().unwrap_or(0)})
        };
        writeln!(w, "{}", json!({"split": split, "tid": tid, "y": y, "tokens": tokens,
            "up": up.as_ref().map(triple_json), "down": down.iter().map(triple_json).collect::<Vec<_>>()}))?;
    }
    let mut r = std::io::BufWriter::new(std::fs::File::create(format!("{out}.rulebook.tsv"))?);
    for ((g, rel, d), c) in &triples {
        writeln!(r, "{g}\t{rel}\t{d}\t{c}")?;
    }
    Ok(())
}
