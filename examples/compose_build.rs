//! Builds the pairing table for the compositional gate (src/compose.rs) from
//! sense-tagged text, parsed with the engine's own parser: SemCor and the
//! Princeton WordNet Gloss Corpus (both WSD-framework format).
//!
//!   cargo run --release --example compose_build [out = models/compose/pairs.bin]
use scribe::compose::Compose;
use scribe::{semcor, Engine};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "models/compose/pairs.bin".into());
    let e = Engine::load(Path::new("models"))?;
    let keys = semcor::sense_keys(Path::new("data/oewn"), &e.lexicon)?;
    let mut b = Compose::builder();
    for base in ["data/semcor/wsdeval/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor", "data/wngc/wngc"] {
        let sents = semcor::parse(&std::fs::read_to_string(format!("{base}.data.xml"))?, &std::fs::read_to_string(format!("{base}.gold.key.txt"))?);
        let (mut arcs, mut tagged) = (0usize, 0usize);
        for (n, s) in sents.iter().enumerate() {
            if s.words.is_empty() || s.words.len() > 80 { continue }
            let mut meaning: Vec<Option<u32>> = vec![None; s.words.len()];
            for inst in &s.instances {
                meaning[inst.index] = inst.keys.iter().find_map(|k| keys.get(k).copied());
            }
            tagged += meaning.iter().flatten().count();
            let parsed = e.parse_tokens(&s.words);
            for (d, t) in parsed.tokens.iter().enumerate() {
                if let (Some(h), Some(md)) = (t.head, meaning[d]) {
                    if let Some(mh) = meaning[h] {
                        b.add_arc(mh, md, &t.deprel);
                        arcs += 1;
                    }
                }
            }
            if n % 20000 == 0 { eprintln!("{base}: {n}/{} sentences", sents.len()) }
        }
        println!("{base}: {} sentences, {tagged} tagged words, {arcs} tagged arcs", sents.len());
    }
    let c = b.finish();
    std::fs::create_dir_all(Path::new(&out).parent().unwrap())?;
    c.save(Path::new(&out))?;
    println!("{out}: {} (meaning, role) entries, {} roles", c.entries.len(), c.roles.len());
    Ok(())
}
