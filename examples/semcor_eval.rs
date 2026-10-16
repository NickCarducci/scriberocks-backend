//! Sense choice on natural, sense-tagged text.
//!
//!   cargo run --release --example semcor_eval [eval-sentences] [train-sentences]
//!
//! SemCor documents are split even/odd: odd documents are evaluated, even
//! ones supply extra sense examples. The Raganato "ALL" set (Senseval/SemEval)
//! is evaluated too. Each set is scored with the shipped sense index (dictionary
//! + Wiktionary examples) and with SemCor examples added, at several margins.
//!
//! Note: WordNet's sense order comes from SemCor counts, so first-listed is
//! unusually strong on SemCor itself; ALL is the fairer comparison.
use scribe::engine::choose_sense;
use scribe::kgram::{self, Example, KgramSenses};
use scribe::semcor::{self, TaggedSentence};
use scribe::usage::senses::SenseIndex;
use scribe::{Engine, Lexicon, Pos};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

const WSD: &str = "data/semcor/wsdeval/WSD_Evaluation_Framework";
const MARGINS: &[f32] = &[0.0, 0.05, 0.1, 0.2, 0.3, 0.5];

/// One evaluated occurrence: its candidate senses (first-listed first), the
/// gold senses and the masked-context vector.
struct Case {
    senses: Vec<u32>,
    gold: Vec<u32>,
    vector: Vec<f32>,
    lemma: String,
    pos: Pos,
    features: Vec<String>,
}

fn load(name: &str, dir: &str) -> Vec<TaggedSentence> {
    let base = format!("{WSD}/{dir}/{name}");
    semcor::parse(&std::fs::read_to_string(format!("{base}.data.xml")).unwrap(), &std::fs::read_to_string(format!("{base}.gold.key.txt")).unwrap())
}

/// Masked-context vectors for each instance, one sentence at a time (the
/// model chunks internally, so memory stays bounded).
fn vectors(e: &Engine, s: &TaggedSentence) -> Vec<Option<Vec<f32>>> {
    let lower: Vec<String> = s.words.iter().map(|w| w.to_lowercase()).collect();
    let masked: Vec<Vec<String>> = s.instances.iter().map(|i| { let mut m = lower.clone(); m[i.index] = "[MASK]".into(); m }).collect();
    match e.usage.as_ref().unwrap().contextual_batch(&masked) {
        Ok(batch) => s.instances.iter().zip(batch).map(|(i, h)| h.get(i.index).cloned()).collect(),
        Err(_) => vec![None; s.instances.len()],
    }
}

fn cases(e: &Engine, keys: &HashMap<String, u32>, sentences: &[TaggedSentence]) -> (Vec<Case>, usize) {
    let (mut out, mut skipped) = (Vec::new(), 0);
    for (n, s) in sentences.iter().enumerate() {
        if n % 500 == 0 { eprintln!("  {n}/{}", sentences.len()) }
        let lower: Vec<String> = s.words.iter().map(|w| w.to_lowercase()).collect();
        for (inst, v) in s.instances.iter().zip(vectors(e, s)) {
            let senses = e.lexicon.senses(&inst.lemma, inst.pos).to_vec();
            let gold: Vec<u32> = inst.keys.iter().filter_map(|k| keys.get(k).copied()).filter(|g| senses.contains(g)).collect();
            match v {
                Some(vector) if !gold.is_empty() && senses.len() > 1 => {
                    out.push(Case { senses, gold, vector, lemma: inst.lemma.clone(), pos: inst.pos, features: kgram::features(&lower, inst.index) })
                }
                _ => skipped += 1,
            }
        }
    }
    (out, skipped)
}

fn report(label: &str, cases: &[Case], index: &SenseIndex, lex: &Lexicon) {
    let n = cases.len().max(1) as f64;
    let first = cases.iter().filter(|c| c.gold.contains(&c.senses[0])).count();
    println!("{label}: {} ambiguous words; first-listed {:.1}%", cases.len(), 100.0 * first as f64 / n);
    for &m in MARGINS {
        let (mut right, mut overrides) = (0, 0);
        for c in cases {
            let (s, _) = choose_sense(&c.senses, |s| index.score(s, &c.vector, lex), m).unwrap();
            right += c.gold.contains(&s) as usize;
            overrides += (s != c.senses[0]) as usize;
        }
        println!("  margin {m:<4}: {:.1}% correct, overrides first-listed {:.1}%", 100.0 * right as f64 / n, 100.0 * overrides as f64 / n);
    }
}

fn report_kgram(label: &str, cases: &[Case], head: &KgramSenses, index: &SenseIndex, lex: &Lexicon) {
    let n = cases.len().max(1) as f64;
    let known = cases.iter().filter(|c| head.knows(&c.lemma, c.pos)).count();
    println!("{label}: k-gram head (lemma seen in training for {:.1}%)", 100.0 * known as f64 / n);
    for m in [2.0f32, 4.0, 6.0, 8.0, 12.0, 16.0] {
        let (mut alone, mut combined, mut overrides) = (0, 0, 0);
        for c in cases {
            let k = head.choose(&c.lemma, c.pos, &c.senses, &c.features, m).unwrap();
            alone += c.gold.contains(&k) as usize;
            overrides += (k != c.senses[0]) as usize;
            let s = match head.knows(&c.lemma, c.pos) {
                true => k,
                false => choose_sense(&c.senses, |s| index.score(s, &c.vector, lex), 0.3).unwrap().0,
            };
            combined += c.gold.contains(&s) as usize;
        }
        println!("  margin {m:<3}: {:.1}% (overrides {:.1}%); with usage vectors for unseen lemmas {:.1}%", 100.0 * alone as f64 / n, 100.0 * overrides as f64 / n, 100.0 * combined as f64 / n);
    }
}

fn main() {
    let arg = |i: usize, d: usize| std::env::args().nth(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (eval_cap, train_cap) = (arg(1, 2000), arg(2, 8000));
    let e = Engine::load(Path::new("models")).unwrap();
    let keys = semcor::sense_keys(Path::new("data/oewn"), &e.lexicon).unwrap();

    let semcor = load("semcor", "Training_Corpora/SemCor");
    let docs: Vec<&str> = semcor.iter().map(|s| s.document.as_str()).collect::<BTreeSet<_>>().into_iter().collect();
    let odd: BTreeSet<&str> = docs.iter().skip(1).step_by(2).copied().collect();
    let (held, train): (Vec<_>, Vec<_>) = semcor.iter().cloned().partition(|s| odd.contains(s.document.as_str()));
    let every = |v: &[TaggedSentence], cap: usize| -> Vec<TaggedSentence> {
        let step = (v.len() / cap.max(1)).max(1);
        v.iter().step_by(step).take(cap).cloned().collect()
    };
    let all_keys: usize = semcor.iter().flat_map(|s| &s.instances).count();
    let mapped: usize = semcor.iter().flat_map(|s| &s.instances).filter(|i| i.keys.iter().any(|k| keys.contains_key(k))).count();
    eprintln!("SemCor: {} sentences in {} documents; {mapped}/{all_keys} sense keys map to the lexicon", semcor.len(), docs.len());

    eprintln!("vectors: SemCor held-out documents");
    let (held_cases, skipped) = cases(&e, &keys, &every(&held, eval_cap));
    eprintln!("  ({skipped} unambiguous or unmapped skipped)");
    eprintln!("vectors: ALL (Senseval/SemEval)");
    let (all_cases, _) = cases(&e, &keys, &every(&load("ALL", "Evaluation_Datasets/ALL"), eval_cap));

    let shipped = e.senses.as_ref().unwrap();
    report("SemCor held-out, shipped index", &held_cases, shipped, &e.lexicon);
    report("ALL, shipped index", &all_cases, shipped, &e.lexicon);

    eprintln!("vectors: SemCor training documents");
    let mut index: SenseIndex = postcard::from_bytes(&std::fs::read("models/usage/senses.bin").unwrap()).unwrap();
    let (train_cases, _) = cases(&e, &keys, &every(&train, train_cap));
    for c in &train_cases {
        index.add(c.gold[0], &c.vector);
    }
    let examples: Vec<Example> = train_cases.iter().map(|c| Example { lemma: c.lemma.clone(), pos: c.pos, sense: c.gold[0], features: c.features.clone() }).collect();
    let head = KgramSenses::train(&examples, 5);
    index.build_backoff(&e.lexicon);
    println!("(+ {} SemCor examples from {} training sentences)", train_cases.len(), train_cap.min(train.len()));
    report("SemCor held-out, + SemCor examples", &held_cases, &index, &e.lexicon);
    report("ALL, + SemCor examples", &all_cases, &index, &e.lexicon);
    report_kgram("SemCor held-out", &held_cases, &head, &index, &e.lexicon);
    report_kgram("ALL", &all_cases, &head, &index, &e.lexicon);
}
