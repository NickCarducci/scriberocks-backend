//! How well a judge's per-word likelihood tells good rocks from editable ones
//! (scripts/fidelity_quality_eval.py → quality_candidates.tsv: label, text).
//! AUC: how often a label-1 text scores higher than a label-0 one (0.5 chance).
//!
//!   cargo run --release --example text_judge_eval [tsv] [judge = models/judges/casual]
use scribe::bert::Bert;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tsv = std::env::args().nth(1).unwrap_or_else(|| "data/fidelity-edits/quality_candidates.tsv".into());
    let judge = std::env::args().nth(2).unwrap_or_else(|| "models/judges/casual".into());
    let bert = Bert::load(Path::new(&judge), scribe::usage::model::Device::Cpu)?;
    let mut scored: Vec<(bool, f64)> = Vec::new();
    for line in std::fs::read_to_string(&tsv)?.lines() {
        let Some((y, text)) = line.split_once('\t') else { continue };
        let words: Vec<String> = scribe::text::tokenize(text).into_iter().map(|w| w.to_lowercase()).collect();
        let s: f64 = (0..words.len()).map(|i| bert.slot_scores(&words, i, std::slice::from_ref(&words[i])).map_or(-20.0, |v| v[0].max(-20.0) as f64)).sum();
        scored.push((y == "1", s / words.len().max(1) as f64));
    }
    let (p, q): (Vec<f64>, Vec<f64>) = (scored.iter().filter(|x| x.0).map(|x| x.1).collect(), scored.iter().filter(|x| !x.0).map(|x| x.1).collect());
    let wins: f64 = p.iter().map(|a| q.iter().map(|b| if a > b { 1.0 } else if a == b { 0.5 } else { 0.0 }).sum::<f64>()).sum();
    println!("{judge}: AUC {:.3} ({} good, {} editable)", wins / (p.len() * q.len()).max(1) as f64, p.len(), q.len());
    Ok(())
}
