//! Meaning-picking end to end: the engine tags, lemmatizes and picks senses
//! itself on the ALL test set (Word Sense Disambiguation Evaluation
//! Framework), instead of being handed the test's lemma and part of speech.
//! Same targets as semcor_eval's ALL score (ambiguous, gold sense in the
//! lexicon), so the two compare directly.
//!
//!   cargo run --release --example sense_pipeline_eval [syntax-dir]
//!
//! Scores the shipped perceptron tagger, then (given a directory saved by
//! parse_finetune) the neural syntax model's tags.
use scribe::semcor::{self, TaggedSentence};
use scribe::syntax::SyntaxModel;
use scribe::Engine;
use std::collections::HashMap;
use std::path::Path;

const WSD: &str = "data/semcor/wsdeval/WSD_Evaluation_Framework";

fn report(label: &str, e: &Engine, keys: &HashMap<String, u32>, all: &[TaggedSentence]) {
    let (mut n, mut pos_right, mut right) = (0usize, 0usize, 0usize);
    for s in all {
        let parsed = e.parse_tokens(&s.words);
        for inst in &s.instances {
            let senses = e.lexicon.senses(&inst.lemma, inst.pos);
            let gold: Vec<u32> = inst.keys.iter().filter_map(|k| keys.get(k).copied()).filter(|g| senses.contains(g)).collect();
            if gold.is_empty() || senses.len() < 2 { continue }
            n += 1;
            let t = &parsed.tokens[inst.index];
            pos_right += (scribe::Pos::from_upos(&t.upos) == Some(inst.pos) && t.lemma == inst.lemma) as usize;
            right += t.sense.is_some_and(|x| gold.contains(&x)) as usize;
        }
    }
    let pct = |x: usize| 100.0 * x as f64 / n.max(1) as f64;
    println!("{label}: meanings {:.1}%  (lemma and part of speech right for {:.1}%; n={n})", pct(right), pct(pos_right));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut e = Engine::load(Path::new("models"))?;
    let keys = semcor::sense_keys(Path::new("data/oewn"), &e.lexicon)?;
    let base = format!("{WSD}/Evaluation_Datasets/ALL/ALL");
    let all = semcor::parse(&std::fs::read_to_string(format!("{base}.data.xml"))?, &std::fs::read_to_string(format!("{base}.gold.key.txt"))?);
    let syntax = e.syntax.take();
    report("perceptron tagger", &e, &keys, &all);
    match std::env::args().nth(1).map(|d| SyntaxModel::load(Path::new(&d))).or(syntax.map(Ok)) {
        Some(m) => {
            e.syntax = Some(m?);
            report("neural syntax", &e, &keys, &all);
        }
        None => println!("(no syntax model given)"),
    }
    Ok(())
}
