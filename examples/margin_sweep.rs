//! How often each override margin replaces the first-listed sense on natural
//! text (EWT test sentences), with a few examples to eyeball. Paired with the
//! held-out accuracy curve from `scribe build-senses`.
use scribe::{conllu, Engine};
use std::path::Path;

fn main() {
    let mut e = Engine::load(Path::new("models")).unwrap();
    let test = conllu::parse(&std::fs::read_to_string("data/en_ewt-ud-test.conllu").unwrap());
    let sentences: Vec<String> = test.iter().take(400).map(|s| s.iter().map(|t| t.form.as_str()).collect::<Vec<_>>().join(" ")).collect();
    for margin in [0.05f32, 0.1, 0.2, 0.3, 0.5] {
        e.sense_margin = margin;
        let (mut words, mut overridden) = (0usize, 0usize);
        let mut shown = 0;
        for text in &sentences {
            for s in e.parse(text) {
                for t in &s.tokens {
                    let Some(sense) = t.sense else { continue };
                    let pos = e.lexicon.synsets[sense as usize].pos;
                    let all = e.lexicon.senses(&t.lemma, pos);
                    if all.len() < 2 { continue }
                    words += 1;
                    if all[0] != sense {
                        overridden += 1;
                        if margin == 0.3 && shown < 12 {
                            shown += 1;
                            println!("    {:<12} → {}", t.form, e.lexicon.synsets[sense as usize].definition);
                        }
                    }
                }
            }
        }
        println!("margin {margin}: overrides first-listed on {:.1}% of {words} ambiguous words", 100.0 * overridden as f64 / words.max(1) as f64);
    }
}
