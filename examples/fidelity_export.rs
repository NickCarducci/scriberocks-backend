//! Exports a FidelityAI sense pilot: for the ambiguous lemmas most frequent
//! in the ALL benchmark, one `sequence_tokens` source file per sense (the
//! ±3-word context of each SemCor training occurrence, as word ids), plus the
//! ALL test cases to score against every sense model.
//!
//!   cargo run --release --example fidelity_export [lemmas] [min-examples-per-sense]
//!
//! Writes data/fidelity/: vocab.txt, <lemma>.<pos>.<synset>.jsonl, tests.jsonl.
//! Only integers leave the machine.
use scribe::semcor::{self, TaggedSentence};
use scribe::{Lexicon, Pos};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

const WSD: &str = "data/semcor/wsdeval/WSD_Evaluation_Framework";
const WINDOW: i32 = 3;

fn load(name: &str, dir: &str) -> Vec<TaggedSentence> {
    let base = format!("{WSD}/{dir}/{name}");
    semcor::parse(&std::fs::read_to_string(format!("{base}.data.xml")).unwrap(), &std::fs::read_to_string(format!("{base}.gold.key.txt")).unwrap())
}

/// Context window as word ids; 0 = sentence edge, 1 = the masked word, 2 = unknown.
fn window(s: &TaggedSentence, i: usize, vocab: &HashMap<String, u32>) -> Vec<u32> {
    (-WINDOW..=WINDOW)
        .map(|k| {
            let j = i as i32 + k;
            if k == 0 { 1 } else if j < 0 || j as usize >= s.words.len() { 0 } else { *vocab.get(&s.words[j as usize].to_lowercase()).unwrap_or(&2) }
        })
        .collect()
}

fn pos_name(p: Pos) -> &'static str {
    match p { Pos::Noun => "n", Pos::Verb => "v", Pos::Adj => "a", Pos::Adv => "r" }
}

fn main() {
    let arg = |i: usize, d: usize| std::env::args().nth(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (n_lemmas, min_examples) = (arg(1, 10), arg(2, 20));
    let lex = Lexicon::load(Path::new("models/lexicon.bin")).unwrap();
    let keys = semcor::sense_keys(Path::new("data/oewn"), &lex).unwrap();
    let train = load("semcor", "Training_Corpora/SemCor");
    let test = load("ALL", "Evaluation_Datasets/ALL");

    // Vocabulary from SemCor words seen at least twice (ids from 3).
    let mut counts: HashMap<String, u32> = HashMap::new();
    for s in &train { for w in &s.words { *counts.entry(w.to_lowercase()).or_default() += 1 } }
    let mut words: Vec<String> = counts.into_iter().filter(|(_, c)| *c >= 2).map(|(w, _)| w).collect();
    words.sort();
    let vocab: HashMap<String, u32> = words.iter().enumerate().map(|(i, w)| (w.clone(), i as u32 + 3)).collect();

    let gold = |keys_: &[String]| -> Option<u32> { keys_.iter().find_map(|k| keys.get(k).copied()) };
    // Training contexts per (lemma, pos) per sense.
    let mut by_sense: HashMap<(String, Pos), HashMap<u32, Vec<Vec<u32>>>> = HashMap::new();
    for s in &train {
        for inst in &s.instances {
            if let Some(g) = gold(&inst.keys) {
                by_sense.entry((inst.lemma.clone(), inst.pos)).or_default().entry(g).or_default().push(window(s, inst.index, &vocab));
            }
        }
    }
    // Most frequent ALL lemmas with at least two senses well represented in SemCor.
    let mut freq: HashMap<(String, Pos), usize> = HashMap::new();
    for s in &test { for inst in &s.instances { *freq.entry((inst.lemma.clone(), inst.pos)).or_default() += 1 } }
    let mut chosen: Vec<((String, Pos), usize)> = freq
        .into_iter()
        .filter(|(k, _)| by_sense.get(k).is_some_and(|m| m.values().filter(|v| v.len() >= min_examples).count() >= 2))
        .collect();
    chosen.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.0.cmp(&b.0.0)));
    chosen.truncate(n_lemmas);

    let out = Path::new("data/fidelity");
    std::fs::create_dir_all(out).unwrap();
    std::fs::write(out.join("vocab.txt"), words.join("\n")).unwrap();
    let mut tests = std::fs::File::create(out.join("tests.jsonl")).unwrap();
    let (mut files, mut bytes) = (0, 0);
    for ((lemma, pos), n_test) in &chosen {
        let senses: Vec<(u32, usize)> = by_sense[&(lemma.clone(), *pos)].iter().filter(|(_, v)| v.len() >= min_examples).map(|(s, v)| (*s, v.len())).collect();
        eprintln!("{lemma}.{}: {n_test} test cases; senses {:?}", pos_name(*pos), senses.iter().map(|s| s.1).collect::<Vec<_>>());
        for (sense, _) in &senses {
            let body: String = by_sense[&(lemma.clone(), *pos)][sense].iter().map(|w| format!("{{\"sequence\":{w:?}}}\n")).collect();
            bytes += body.len();
            files += 1;
            std::fs::write(out.join(format!("{lemma}.{}.{sense}.jsonl", pos_name(*pos))), body).unwrap();
        }
        let first = lex.senses(lemma, *pos).first().copied();
        for s in &test {
            for inst in s.instances.iter().filter(|i| &i.lemma == lemma && i.pos == *pos) {
                let g: Vec<u32> = inst.keys.iter().filter_map(|k| keys.get(k).copied()).collect();
                let modeled: Vec<u32> = senses.iter().map(|s| s.0).collect();
                writeln!(tests, "{{\"lemma\":\"{lemma}\",\"pos\":\"{}\",\"gold\":{g:?},\"first\":{},\"senses\":{modeled:?},\"sequence\":{:?}}}", pos_name(*pos), first.map_or(-1, |f| f as i64), window(s, inst.index, &vocab)).unwrap();
            }
        }
    }
    eprintln!("{files} sense models, {:.1} KB of source in total; vocabulary {}", bytes as f64 / 1e3, words.len());
}
