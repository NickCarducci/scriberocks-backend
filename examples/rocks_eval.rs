//! ScribeSLM's standing acceptance rocks (tests/rocks.tsv): for each rock,
//! the suggestions it must offer and must never offer. Reports every miss
//! and every forbidden suggestion, so models are compared on the same rocks
//! alongside SWORDS and IteraTeR.
//!
//!   cargo run --release --example rocks_eval [models-dir] [suggestions per rock]
use scribe::Engine;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "models".into());
    let limit: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(12);
    let e = Engine::load(Path::new(&dir))?;
    let (mut offered, mut wanted, mut forbidden, mut banned) = (0, 0, 0, 0);
    for line in std::fs::read_to_string("tests/rocks.tsv")?.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        let rock = cols[0];
        let list = |i: usize| -> Vec<&str> { cols.get(i).map(|c| c.split('|').filter(|x| !x.is_empty()).collect()).unwrap_or_default() };
        let texts: Vec<String> = e.variants(rock, limit).into_iter().map(|v| v.text.to_lowercase()).collect();
        let has = |x: &str| texts.iter().any(|t| t.contains(&x.to_lowercase()));
        println!("{rock}");
        for x in list(1) {
            wanted += 1;
            if has(x) { offered += 1 } else { println!("   missing   {x}") }
        }
        for x in list(2) {
            banned += 1;
            if has(x) { forbidden += 1; println!("   forbidden {x}") }
        }
    }
    println!("acceptance rocks: offered {offered}/{wanted} wanted suggestions; {forbidden}/{banned} forbidden ones appeared (top {limit} per rock)");
    Ok(())
}
