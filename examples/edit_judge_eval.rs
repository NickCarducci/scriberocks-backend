//! How well Scriberocks' own models tell a real editor's edit from the same
//! edit displaced (scripts/fidelity_edit_records.py → test_sentences.tsv):
//! AUC of the change in score from before to after the edit (0.5 chance).
//! The bar FidelityAI's typicality model has to beat to rank edits.
//!
//!   cargo run --release --example edit_judge_eval [tsv = data/fidelity-edits/test_sentences.tsv] [judge = models/judges/casual]
use scribe::bert::Bert;
use scribe::Engine;
use std::path::Path;

/// Mean log-probability per word with each word hidden in turn.
fn pll(bert: &Bert, words: &[String]) -> f64 {
    let s: f64 = (0..words.len()).map(|i| bert.slot_scores(words, i, std::slice::from_ref(&words[i])).map_or(-20.0, |v| v[0].max(-20.0) as f64)).sum();
    s / words.len().max(1) as f64
}

fn auc(scores: &[(bool, f64)]) -> f64 {
    let (pos, neg): (Vec<f64>, Vec<f64>) = (scores.iter().filter(|s| s.0).map(|s| s.1).collect(), scores.iter().filter(|s| !s.0).map(|s| s.1).collect());
    let wins: f64 = pos.iter().map(|p| neg.iter().map(|n| if p > n { 1.0 } else if p == n { 0.5 } else { 0.0 }).sum::<f64>()).sum();
    wins / (pos.len() * neg.len()).max(1) as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tsv = std::env::args().nth(1).unwrap_or_else(|| "data/fidelity-edits/test_sentences.tsv".into());
    let judge = std::env::args().nth(2).unwrap_or_else(|| "models/judges/casual".into());
    let e = Engine::load(Path::new("models"))?;
    let bert = Bert::load(Path::new(&judge), scribe::usage::model::Device::Cpu)?;
    let (mut lm, mut pl) = (Vec::new(), Vec::new());
    for line in std::fs::read_to_string(&tsv)?.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() != 3 { continue }
        let real = c[0] == "1";
        let (a, b): (Vec<String>, Vec<String>) = (c[1].split(' ').map(String::from).collect(), c[2].split(' ').map(String::from).collect());
        lm.push((real, e.fluency(&b) - e.fluency(&a)));
        pl.push((real, pll(&bert, &b) - pll(&bert, &a)));
    }
    println!("{} edits ({} real, {} displaced)", lm.len(), lm.iter().filter(|x| x.0).count(), lm.iter().filter(|x| !x.0).count());
    println!("  trigram fluency change (ScribeSLM's ordering today): AUC {:.3}", auc(&lm));
    println!("  {judge} judge, per-word likelihood change:           AUC {:.3}", auc(&pl));
    Ok(())
}
