//! Suggestions for a rock in each available prose style, side by side.
//! Judges load from models/judges/<style>/ (BERT-format model folders).
//!
//!   echo "your rock" | cargo run --release --example style_demo [limit]
//! Env: VARIETY=0..1 (the app's slider), STYLE=casual|publishable (just that one).
use scribe::engine::ProseStyle;
use scribe::scribe_ie::EditOptions;
use scribe::Engine;
use std::io::Read;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limit: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(6);
    let e = Engine::load(Path::new("models"))?;
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
        for v in e.variants_with(rock.trim(), limit, &EditOptions { style, variety, ..Default::default() }) {
            println!("   {:<16} {}", v.ops.iter().map(|o| format!("{o:?}").to_lowercase()).collect::<Vec<_>>().join("+"), v.text);
        }
    }
    Ok(())
}
