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
    // Optional third column form "group\tlabel\ttext": also report, within each
    // group, how often a label-1 text beats a label-0 one (VISIBLE=<groups file> filters).
    let visible: Option<std::collections::HashSet<String>> = std::env::var("VISIBLE").ok().filter(|p| !p.is_empty()).map(|p| std::fs::read_to_string(p).unwrap_or_default().split_whitespace().map(String::from).collect());
    let mut groups: std::collections::HashMap<String, (Vec<f64>, Vec<f64>)> = std::collections::HashMap::new();
    let mut scored: Vec<(bool, f64)> = Vec::new();
    for line in std::fs::read_to_string(&tsv)?.lines() {
        let cols: Vec<&str> = line.splitn(3, '\t').collect();
        let (group, y, text) = match cols.as_slice() { [g, y, t] => (Some(*g), *y, *t), [y, t] => (None, *y, *t), _ => continue };
        if let (Some(v), Some(g)) = (&visible, group) { if !v.contains(g) { continue } }
        let words: Vec<String> = scribe::text::tokenize(text).into_iter().map(|w| w.to_lowercase()).collect();
        let s: f64 = (0..words.len()).map(|i| bert.slot_scores(&words, i, std::slice::from_ref(&words[i])).map_or(-20.0, |v| v[0].max(-20.0) as f64)).sum();
        let score = s / words.len().max(1) as f64;
        scored.push((y == "1", score));
        if let Some(g) = group {
            let e = groups.entry(g.to_string()).or_default();
            if y == "1" { e.0.push(score) } else { e.1.push(score) }
        }
    }
    let (p, q): (Vec<f64>, Vec<f64>) = (scored.iter().filter(|x| x.0).map(|x| x.1).collect(), scored.iter().filter(|x| !x.0).map(|x| x.1).collect());
    let wins: f64 = p.iter().map(|a| q.iter().map(|b| if a > b { 1.0 } else if a == b { 0.5 } else { 0.0 }).sum::<f64>()).sum();
    println!("{judge}: AUC {:.3} ({} good, {} editable)", wins / (p.len() * q.len()).max(1) as f64, p.len(), q.len());
    if !groups.is_empty() {
        let (mut w, mut n) = (0.0, 0usize);
        for (a, r) in groups.values() {
            for x in a { for y in r { w += if x > y { 1.0 } else if x == y { 0.5 } else { 0.0 }; n += 1 } }
        }
        println!("{judge}: within group, label 1 beats label 0 {:.3} of {n} pairs", w / n.max(1) as f64);
    }
    Ok(())
}
