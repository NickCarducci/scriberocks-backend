//! Fill-in-the-blank on held-out everyday text that matches ScribeSLM's
//! standards corpus (none of it trained on): EWT test sentences (everyday web
//! English), Tatoeba's 2,000 held-out learner sentences, and WordNet's
//! held-out example sentences. Each word is hidden in turn; scored by whether
//! the model's top 1 / top 10 guesses contain it. Only words that every model
//! compared stores as a single vocabulary piece count, so different
//! vocabularies compete on the same words.
//!
//!   cargo run --release --example fill_eval <BERT-format model dir>... [--limit 400] [--gum]
//!
//! --gum: instead, a DIFFERENT corpus split by subdomain — GUM (Georgetown
//! University Multilayer corpus) dev+test, its CC-BY / public-domain genres
//! only (wikiHow and fiction are non-commercial; others unconfirmed): casual's
//! domain (conversation, vlog, interview, speech) and publishable's (news, bio,
//! voyage, textbook). Test data only.
use scribe::bert::Bert;
use scribe::usage::corpus;
use scribe::{conllu, text, Lexicon};
use std::path::Path;

/// GUM sentences (lowercased forms) by genre, permissive genres only.
const GUM_GENRES: [(&str, &str); 8] = [
    ("conversation", "casual"), ("vlog", "casual"), ("interview", "casual"), ("speech", "casual"),
    ("news", "publishable"), ("bio", "publishable"), ("voyage", "publishable"), ("textbook", "publishable"),
];

fn gum_sets(limit: usize) -> Result<Vec<(&'static str, Vec<Vec<String>>)>, Box<dyn std::error::Error>> {
    let mut by: std::collections::HashMap<String, Vec<Vec<String>>> = std::collections::HashMap::new();
    for f in ["data/gum/en_gum-ud-dev.conllu", "data/gum/en_gum-ud-test.conllu"] {
        let (mut genre, mut words) = (String::new(), Vec::new());
        for line in std::fs::read_to_string(f)?.lines() {
            if let Some(g) = line.strip_prefix("# meta::genre = ") { genre = g.trim().to_string() }
            else if line.is_empty() {
                if !words.is_empty() { by.entry(genre.clone()).or_default().push(std::mem::take(&mut words)) }
            } else if !line.starts_with('#') {
                let cols: Vec<&str> = line.split('\t').collect();
                if cols.len() > 1 && cols[0].chars().all(|c| c.is_ascii_digit()) { words.push(cols[1].to_lowercase()) }
            }
        }
    }
    let labels: Vec<(&'static str, Vec<Vec<String>>)> = GUM_GENRES.iter().map(|&(g, domain)| {
        let v = by.remove(g).unwrap_or_default();
        let step = (v.len() / limit.max(1)).max(1);
        let name: &'static str = Box::leak(format!("GUM {g} ({domain}'s domain)").into_boxed_str());
        (name, v.into_iter().step_by(step).take(limit).collect())
    }).collect();
    Ok(labels)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let limit: usize = args.windows(2).find(|w| w[0] == "--limit").and_then(|w| w[1].parse().ok()).unwrap_or(400);
    let dirs: Vec<&String> = args.iter().enumerate().filter(|(i, a)| !a.starts_with("--") && (*i == 0 || args[i - 1] != "--limit")).map(|(_, a)| a).collect();
    let models: Vec<(String, Bert)> = dirs.iter().map(|d| Ok((d.to_string(), Bert::load(Path::new(d.as_str()), scribe::usage::model::Device::Cpu)?))).collect::<Result<_, Box<dyn std::error::Error>>>()?;
    let lex = Lexicon::load(Path::new("models/lexicon.bin"))?;
    let gum = args.iter().any(|a| a == "--gum");
    let lower = |w: Vec<String>| -> Vec<String> { w.into_iter().map(|x| x.to_lowercase()).collect() };
    let spread = |v: Vec<Vec<String>>| -> Vec<Vec<String>> { let step = (v.len() / limit).max(1); v.into_iter().step_by(step).take(limit).collect() };
    let sets: Vec<(&str, Vec<Vec<String>>)> = if gum { gum_sets(limit)? } else { vec![
        ("EWT test (everyday web English)", spread(conllu::parse(&std::fs::read_to_string("data/en_ewt-ud-test.conllu")?).iter().map(|s| lower(s.iter().map(|t| t.form.clone()).collect())).collect())),
        ("Tatoeba held-out (learners)", spread(std::fs::read_to_string("data/corpus/tatoeba.txt")?.lines().take(2000).map(|l| lower(text::tokenize(l))).collect())),
        ("WordNet held-out examples (lexicographers)", spread(corpus::wordnet_examples(&lex).into_iter().filter(|e| corpus::is_held_out(e.synset, &e.sentence, 10)).map(|e| lower(text::tokenize(&e.sentence))).collect())),
    ] };
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
