//! Clause frames whose slot values are WordNet meanings, for a FidelityAI
//! rulebook of known sequences: SemCor and the WordNet Gloss Corpus (gold
//! meanings from their annotators), and the SWORDS candidate sentences.
//! Synonyms share a synset, so "car" and "automobile" fill a slot alike.
//!
//! Each frame slot (subject, action, object, target) gets: its lemma; the
//! gold synset (annotated corpora only); the synset the sense head predicts;
//! and each synset's lexicographer category (`noun.artifact`, ...). Modal and
//! status come from the frame. One JSON line per sentence.
//!
//!   cargo run --release --example meaning_frames -- <out.jsonl> [every-nth = 4]
//!
//! The sense head was trained on these corpora, so its predictions on them
//! are better than on unseen text (the SWORDS candidates are unseen).
use scribe::fidelity_frames::{prepare, sentence_frames, GrammarFrame, SlotWords};
use scribe::semcor;
use scribe::Engine;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;

const CORPORA: [(&str, &str); 2] =
    [("semcor", "data/semcor/wsdeval/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor"), ("wngc", "data/wngc/wngc")];

fn frames_json(e: &Engine, words: &[String], gold: &HashMap<usize, u32>) -> Vec<Value> {
    let parsed = e.parse_tokens(words);
    let synset = |s: Option<u32>| s.map(|s| json!({"s": format!("s{s}"), "c": e.lexicon.category_name(s)}));
    sentence_frames(e, words)
        .into_iter()
        .map(|(f, w): (GrammarFrame, SlotWords)| {
            let slot = |lemma: &Option<String>, at: Option<usize>| match (lemma, at) {
                (Some(l), Some(i)) => json!({"lemma": l, "gold": synset(gold.get(&i).copied()), "pred": synset(parsed.tokens[i].sense)}),
                (Some(l), None) => json!({"lemma": l}),
                _ => Value::Null,
            };
            json!({
                "subject": slot(&f.subject, w.subject),
                "action": slot(&f.action, w.action),
                "object": slot(&f.object, w.object),
                "target": slot(&f.target, w.target),
                "modal": f.modal,
                "status": f.status,
            })
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.get(1).ok_or("usage: meaning_frames <out.jsonl> [every-nth]")?;
    let every: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(4);
    let mut e = Engine::load(Path::new("models"))?;
    prepare(&mut e);
    let keys = semcor::sense_keys(Path::new("data/oewn"), &e.lexicon)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path)?);
    for (name, base) in CORPORA {
        let sentences = semcor::parse(&std::fs::read_to_string(format!("{base}.data.xml"))?, &std::fs::read_to_string(format!("{base}.gold.key.txt"))?);
        let (mut n, mut frames) = (0, 0);
        for (i, s) in sentences.iter().enumerate().filter(|(i, _)| i % every == 0) {
            let gold: HashMap<usize, u32> =
                s.instances.iter().filter_map(|inst| inst.keys.iter().find_map(|k| keys.get(k)).map(|&g| (inst.index, g))).collect();
            let f = frames_json(&e, &s.words, &gold);
            frames += f.len();
            n += 1;
            writeln!(out, "{}", json!({"corpus": name, "doc": s.document, "i": i, "frames": f}))?;
        }
        eprintln!("{name}: {n} sentences, {frames} frames");
    }
    let swords = std::fs::File::open("data/fidelity-edits/swords_candidates.tsv")?;
    let mut n = 0;
    for line in std::io::BufReader::new(swords).lines() {
        let line = line?;
        let mut cols = line.splitn(3, '\t');
        let (Some(tid), Some(y), Some(text)) = (cols.next(), cols.next(), cols.next()) else { continue };
        let frames: Vec<Value> = scribe::text::sentences(text).iter().flat_map(|s| frames_json(&e, &scribe::text::tokenize(s), &HashMap::new())).collect();
        writeln!(out, "{}", json!({"corpus": "swords", "tid": tid, "y": y.parse::<u8>()?, "i": n, "frames": frames}))?;
        n += 1;
    }
    eprintln!("swords: {n} candidates");
    Ok(())
}
