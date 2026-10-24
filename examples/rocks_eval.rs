//! ScribeSLM's standing acceptance rocks (tests/rocks.tsv): for each rock,
//! the suggestions it must offer and must never offer. Reports every miss
//! and every forbidden suggestion, so models are compared on the same rocks
//! alongside SWORDS and IteraTeR.
//!
//!   cargo run --release --example rocks_eval [models-dir] [suggestions per rock]
//!   JUDGE_DIR=<BERT-format model dir> judges swaps with that model instead of the usage model.
//!   ROCKS=<file> (default tests/rocks.tsv). Entries "w:<lemma>" match a word
//!   (by dictionary form) a suggestion has that the rock doesn't; others match
//!   as case-insensitive substrings.
use scribe::Engine;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "models".into());
    let limit: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(12);
    let mut e = Engine::load(Path::new(&dir))?;
    // SWAP_MARGIN / CLOSENESS override the swap filter's cut-off and closeness weight.
    if let Some(m) = std::env::var("SWAP_MARGIN").ok().and_then(|v| v.parse().ok()) { e.swap_margin = m }
    if let Some(w) = std::env::var("CLOSENESS").ok().and_then(|v| v.parse().ok()) { e.closeness_weight = w }
    // COMPOSE=<pairs.bin> turns on the pairing gate (src/compose.rs); GATE_MIN,
    // GATE_TYPICAL and GATE_OUTCOMPETED set it.
    if let Ok(path) = std::env::var("COMPOSE") {
        let f = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
        let mut gate = scribe::compose::Gate::default();
        if let Some(v) = f("GATE_MIN") { gate.min_seen = v as usize }
        if let Some(v) = f("GATE_TYPICAL") { gate.typical = v }
        if let Some(v) = f("GATE_OUTCOMPETED") { gate.outcompeted = v }
        e.compose = Some((scribe::compose::Compose::load(Path::new(&path))?, gate));
    }
    // SWAP_CONFIDENCE=<gap>: swaps only where the sense head's chosen meaning beats its rival by this.
    if let Some(c) = std::env::var("SWAP_CONFIDENCE").ok().and_then(|v| v.parse().ok()) { e.swap_confidence = Some(c) }
    if let Ok(judge) = std::env::var("JUDGE_DIR") {
        e.judge = Some(Box::new(scribe::bert::Bert::load(Path::new(&judge), scribe::usage::model::Device::Cpu)?));
        println!("judge: {judge}");
    }
    // STYLE=casual|publishable judges with that style's model; GENERATE=1 turns
    // ScribeIE on (its ranks order the suggestions).
    let options = scribe::scribe_ie::EditOptions {
        style: std::env::var("STYLE").ok().and_then(|s| scribe::engine::ProseStyle::from_name(&s)),
        generate: std::env::var_os("GENERATE").is_some(),
        ..Default::default()
    };
    let (mut offered, mut wanted, mut forbidden, mut banned) = (0, 0, 0, 0);
    let file = std::env::var("ROCKS").unwrap_or_else(|_| "tests/rocks.tsv".into());
    let lemmas = |t: &str| -> std::collections::HashSet<String> {
        scribe::text::tokenize(t).iter().map(|w| e.lexicon.lemmatize(w, None).into_iter().next().unwrap_or_else(|| w.to_lowercase()).to_lowercase()).collect()
    };
    for line in std::fs::read_to_string(&file)?.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        let rock = cols[0];
        let list = |i: usize| -> Vec<&str> { cols.get(i).map(|c| c.split('|').filter(|x| !x.is_empty()).collect()).unwrap_or_default() };
        let texts: Vec<String> = e.variants_with(rock, limit, &options).into_iter().map(|v| v.text.to_lowercase()).collect();
        let rock_lemmas = lemmas(rock);
        let added: Vec<std::collections::HashSet<String>> = texts.iter().map(|t| lemmas(t).difference(&rock_lemmas).cloned().collect()).collect();
        let has = |x: &str| match x.strip_prefix("w:") {
            Some(w) => added.iter().any(|a| a.contains(w)),
            None => texts.iter().any(|t| t.contains(&x.to_lowercase())),
        };
        if std::env::var_os("QUIET").is_none() { println!("{rock}") }
        for x in list(1) {
            wanted += 1;
            if has(x) { offered += 1 } else if std::env::var_os("QUIET").is_none() { println!("   missing   {x}") }
        }
        for x in list(2) {
            banned += 1;
            if has(x) { forbidden += 1; if std::env::var_os("QUIET").is_none() { println!("   forbidden {x}") } }
        }
    }
    println!("acceptance rocks: offered {offered}/{wanted} wanted suggestions; {forbidden}/{banned} forbidden ones appeared (top {limit} per rock)");
    Ok(())
}
