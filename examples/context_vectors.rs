//! A word's vector as a usage model reads it in context: the mean of its contextual
//! vectors across up to n real corpus sentences containing it, as JSON {word: vector}.
//!
//!   cargo run --release --example context_vectors -- <words.txt> <corpus.txt> <model dir> <out.json> [n = 4]
use scribe::bert::Bert;
use std::collections::{HashMap, HashSet};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(5).map(|s| s.parse()).transpose()?.unwrap_or(4);
    let wanted: HashSet<String> = std::fs::read_to_string(&a[1])?.lines().map(str::to_lowercase).collect();
    let bert = Bert::load(Path::new(&a[3]), scribe::usage::model::Device::Cpu)?;
    let mut uses: HashMap<String, usize> = HashMap::new();
    let (mut batch, mut sums): (Vec<Vec<String>>, HashMap<String, (Vec<f32>, usize)>) = (vec![], HashMap::new());
    let mut flush = |batch: &mut Vec<Vec<String>>| -> Result<(), Box<dyn std::error::Error>> {
        for (words, vectors) in batch.iter().zip(bert.contextual_batch(batch)?) {
            for (w, v) in words.iter().zip(vectors) {
                if wanted.contains(w) {
                    let e = sums.entry(w.clone()).or_insert_with(|| (vec![0.0; v.len()], 0));
                    if e.1 < n {
                        e.0.iter_mut().zip(&v).for_each(|(x, y)| *x += y);
                        e.1 += 1;
                    }
                }
            }
        }
        batch.clear();
        Ok(())
    };
    for line in std::fs::read_to_string(&a[2])?.lines() {
        for s in scribe::text::sentences(line) {
            let words: Vec<String> = scribe::text::tokenize(&s).iter().map(|w| w.to_lowercase()).collect();
            if words.len() < 5 || words.len() > 40 {
                continue;
            }
            let mut useful = false;
            for w in &words {
                if wanted.contains(w) && uses.get(w).copied().unwrap_or(0) < n {
                    *uses.entry(w.clone()).or_insert(0) += 1;
                    useful = true;
                }
            }
            if useful {
                batch.push(words);
                if batch.len() == 32 {
                    flush(&mut batch)?;
                }
            }
        }
    }
    flush(&mut batch)?;
    let out: HashMap<String, Vec<f32>> = sums.into_iter().map(|(w, (v, c))| (w, v.iter().map(|x| x / c as f32).collect())).collect();
    eprintln!("{} of {} words found", out.len(), wanted.len());
    std::fs::write(&a[4], serde_json::to_string(&out)?)?;
    Ok(())
}
