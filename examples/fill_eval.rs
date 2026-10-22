//! Fill-in-the-blank on held-out everyday text that matches ScribeSLM's
//! standards corpus (none of it trained on): EWT test sentences (everyday web
//! English), Tatoeba's 2,000 held-out learner sentences, and WordNet's
//! held-out example sentences. Each word is hidden in turn; scored by whether
//! the model's top 1 / top 10 guesses contain it. Only words that every model
//! compared stores as a single vocabulary piece count, so different
//! vocabularies compete on the same words.
//!
//!   cargo run --release --example fill_eval <BERT-format model dir>... [--limit 400]
use scribe::bert::Bert;
use scribe::usage::corpus;
use scribe::{conllu, text, Lexicon};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let limit: usize = args.windows(2).find(|w| w[0] == "--limit").and_then(|w| w[1].parse().ok()).unwrap_or(400);
    let dirs: Vec<&String> = args.iter().enumerate().filter(|(i, a)| !a.starts_with("--") && (*i == 0 || args[i - 1] != "--limit")).map(|(_, a)| a).collect();
    let models: Vec<(String, Bert)> = dirs.iter().map(|d| Ok((d.to_string(), Bert::load(Path::new(d.as_str()), scribe::usage::model::Device::Cpu)?))).collect::<Result<_, Box<dyn std::error::Error>>>()?;
    let lex = Lexicon::load(Path::new("models/lexicon.bin"))?;
    let lower = |w: Vec<String>| -> Vec<String> { w.into_iter().map(|x| x.to_lowercase()).collect() };
    let spread = |v: Vec<Vec<String>>| -> Vec<Vec<String>> { let step = (v.len() / limit).max(1); v.into_iter().step_by(step).take(limit).collect() };
    let sets: Vec<(&str, Vec<Vec<String>>)> = vec![
        ("EWT test (everyday web English)", spread(conllu::parse(&std::fs::read_to_string("data/en_ewt-ud-test.conllu")?).iter().map(|s| lower(s.iter().map(|t| t.form.clone()).collect())).collect())),
        ("Tatoeba held-out (learners)", spread(std::fs::read_to_string("data/corpus/tatoeba.txt")?.lines().take(2000).map(|l| lower(text::tokenize(l))).collect())),
        ("WordNet held-out examples (lexicographers)", spread(corpus::wordnet_examples(&lex).into_iter().filter(|e| corpus::is_held_out(e.synset, &e.sentence, 10)).map(|e| lower(text::tokenize(&e.sentence))).collect())),
    ];
    println!("fill-in-the-blank, word hidden; top 1 / top 10 (words every model stores as one piece)");
    for (name, sentences) in &sets {
        let mut n = 0usize;
        let mut hits = vec![(0usize, 0usize); models.len()];
        for s in sentences {
            for i in 0..s.len() {
                let w = &s[i];
                if !w.chars().all(char::is_alphabetic) || models.iter().any(|(_, m)| m.single_piece(w).is_none()) { continue }
                n += 1;
                for (k, (_, m)) in models.iter().enumerate() {
                    let top: Vec<String> = m.predict(s, i, 10, false).unwrap_or_default().into_iter().map(|x| x.0).collect();
                    hits[k].0 += (top.first() == Some(w)) as usize;
                    hits[k].1 += top.contains(w) as usize;
                }
            }
        }
        println!("{name}: {n} words");
        for (k, (dir, _)) in models.iter().enumerate() {
            println!("  {dir:<32} {:>5.1}% / {:>5.1}%", 100.0 * hits[k].0 as f64 / n.max(1) as f64, 100.0 * hits[k].1 as f64 / n.max(1) as f64);
        }
    }
    Ok(())
}
