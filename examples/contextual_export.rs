//! Exports the usage model's contextual vector at one word of each sentence,
//! for external evaluation (e.g. FidelityAI dense-role experiments).
//!
//!   cargo run --release --example contextual_export <model> <in.jsonl> <out.f32>
//!
//! Input lines: {"words": [...], "at": i}. Output: little-endian f32 rows of
//! the model's dimension, one per input line; a word whose pieces fall past
//! the model's window gets a row of zeros.
use scribe::usage::model::{Device, UsageModel};
use std::io::{BufRead, Write};
use std::path::Path;

#[derive(serde::Deserialize)]
struct Line {
    words: Vec<String>,
    at: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model = UsageModel::load(&Path::new("models").join(&args[1]), Device::Cpu)?;
    let dim = model.config.dim;
    let lines: Vec<Line> = std::io::BufReader::new(std::fs::File::open(&args[2])?)
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[3])?);
    let mut missing = 0;
    for chunk in lines.chunks(256) {
        let sentences: Vec<Vec<String>> = chunk.iter().map(|l| l.words.clone()).collect();
        for (l, h) in chunk.iter().zip(model.contextual_batch(&sentences)?) {
            let row = h.get(l.at).cloned().unwrap_or_else(|| {
                missing += 1;
                vec![0.0; dim]
            });
            for x in row {
                out.write_all(&x.to_le_bytes())?;
            }
        }
    }
    eprintln!("{} vectors of dimension {dim} ({missing} past the window)", lines.len());
    Ok(())
}
