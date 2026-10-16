//! Acceptance tests against the real trained models (core/models).
//! Skipped when the models haven't been built.

use scribe::Engine;
use std::path::Path;

fn engine() -> Option<Engine> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    if !dir.join("usage/usage.safetensors").exists() {
        eprintln!("skipping: build models first (scribe train, scribe train-usage)");
        return None;
    }
    Some(Engine::load(&dir).expect("models load"))
}

fn swaps(engine: &Engine, rock: &str) -> Vec<String> {
    engine.variants(rock, 200).into_iter().filter(|v| v.ops.contains(&scribe::Op::Swap)).map(|v| v.text).collect()
}

#[test]
fn usage_model_drops_swaps_that_dont_fit_the_sentence() {
    let Some(engine) = engine() else { return };
    let got = swaps(&engine, "I'm too tired after work.");
    for bad in ["too gone", "too played out", "too all in"] {
        assert!(!got.iter().any(|v| v.contains(bad)), "{bad:?} offered: {got:#?}");
    }
}

#[test]
fn usage_model_keeps_swaps_that_fit() {
    let Some(engine) = engine() else { return };
    let got = swaps(&engine, "Why am I always afraid to ask for help?");
    assert!(got.iter().any(|v| v.contains("scared") || v.contains("frightened")), "{got:#?}");
}

#[test]
fn engine_reports_that_the_usage_model_is_loaded() {
    let Some(engine) = engine() else { return };
    assert!(engine.has_usage_model());
}

fn connects(engine: &Engine, rock: &str) -> Vec<String> {
    engine.variants(rock, 200).into_iter().filter(|v| v.ops.contains(&scribe::Op::Connect)).map(|v| v.text).collect()
}

#[test]
fn connect_joins_statements_with_a_closed_set_connective() {
    let Some(engine) = engine() else { return };
    let got = connects(&engine, "I stayed home. It rained.");
    assert!(!got.is_empty(), "no connect variants");
    for v in &got {
        let added: Vec<&str> = v.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty())
            .filter(|w| !["I", "stayed", "home", "it", "It", "rained"].contains(w)).collect();
        assert!(added.len() == 1 && scribe::variants::CONNECTIVES.contains(&added[0]), "{v:?} adds {added:?}");
    }
}

#[test]
fn connect_never_joins_questions() {
    let Some(engine) = engine() else { return };
    assert!(connects(&engine, "Why did I stay home? It rained.").is_empty());
}
