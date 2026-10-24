//! ScribeIE's rewrite of each stdin line, at a few confidences.
//!
//!   echo "a sentence" | cargo run --release --example scribe_ie_demo [style = publishable]
use scribe::edit_tagger::{EditTagger, TaggerEnhancer};
use scribe::scribe_ie::Enhancer;
use std::io::Read;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let style = std::env::args().nth(1).unwrap_or_else(|| "publishable".into());
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    for line in input.lines().filter(|l| !l.trim().is_empty()) {
        println!("{line}");
        for confidence in [0.3, 0.5, 0.7] {
            let e = TaggerEnhancer {
                encoder: scribe::bert::Bert::load(&Path::new("models/judges").join(&style), scribe::usage::model::Device::Cpu)?,
                tagger: EditTagger::load(&Path::new("models/scribe-ie").join(&style))?,
                confidence,
            };
            println!("  {confidence:.1}: {}", e.enhance(line, line).unwrap_or_else(|| "(unchanged)".into()));
        }
    }
    Ok(())
}
