//! Suggestions for a rock in each available prose style, side by side.
//! Judges load from models/judges/<style>/ (BERT-format model folders).
//!
//!   echo "your rock" | cargo run --release --example style_demo [limit]
//! Env: VARIETY=0..1 (the app's slider), STYLE=casual|publishable (just that one),
//! COMPOSE=<pairs.bin> with GATE_TYPICAL / GATE_OUTCOMPETED (the pairing gate),
//! GENERATE=1 (ScribeIE: ranks the suggestions and writes red versions).
use scribe::engine::ProseStyle;
use scribe::scribe_ie::EditOptions;
use scribe::Engine;
use std::io::Read;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limit: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(6);
    let mut e = Engine::load(Path::new("models"))?;
    if let Some(c) = std::env::var("SWAP_CONFIDENCE").ok().and_then(|v| v.parse().ok()) { e.swap_confidence = Some(c) }
    // COMPOSE=<pairs.bin> + GATE_TYPICAL / GATE_OUTCOMPETED: the pairing gate (src/compose.rs).
    if let Ok(path) = std::env::var("COMPOSE") {
        let f = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok());
        let gate = scribe::compose::Gate { typical: f("GATE_TYPICAL").unwrap_or(0.0), outcompeted: f("GATE_OUTCOMPETED").unwrap_or(0.0), ..Default::default() };
        e.compose = Some((scribe::compose::Compose::load(Path::new(&path))?, gate));
    }
    let mut rock = String::new();
    std::io::stdin().read_to_string(&mut rock)?;
    let variety = std::env::var("VARIETY").ok().and_then(|v| v.parse().ok());
    let only = std::env::var("STYLE").ok().and_then(|s| ProseStyle::from_name(&s));
    let styles: Vec<Option<ProseStyle>> = match only {
        Some(s) => vec![Some(s)],
        None => std::iter::once(None).chain(e.available_styles().into_iter().map(Some)).collect(),
    };
    for style in styles {
        println!("== {}", style.map_or("default judge (live usage model)", |s| s.name()));
        for v in e.variants_with(rock.trim(), limit, &EditOptions { style, variety, generate: std::env::var_os("GENERATE").is_some() }) {
            println!("   {:<16} {}", v.ops.iter().map(|o| format!("{o:?}").to_lowercase()).collect::<Vec<_>>().join("+"), v.text);
            if let Some(e) = &v.enhanced { println!("   {:<16} {}", "  ScribeIE →", e) }
        }
    }
    Ok(())
}
