//! Acceptance tests against the real trained models (core/models).
//! Skipped when the models haven't been built.

use scribe::Engine;
use std::path::Path;

fn engine() -> Option<Engine> {
    // SCRIBE_MODELS points at another models dir (e.g. to compare usage models).
    let dir = std::env::var("SCRIBE_MODELS").map(std::path::PathBuf::from).unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("models"));
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

fn punctuated(engine: &Engine, rock: &str) -> Vec<String> {
    engine.variants(rock, 200).into_iter().filter(|v| v.ops == vec![scribe::Op::Punctuate]).map(|v| v.text).collect()
}

fn letters(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric() || c.is_whitespace()).collect::<String>().to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

#[test]
fn punctuate_adds_commas_and_end_marks_where_they_belong() {
    let Some(engine) = engine() else { return };
    for (rock, want) in [
        ("Honestly I need a break", "Honestly, I need a break."),
        ("I was tired so I went to bed", "I was tired, so I went to bed."),
        ("where are my keys", "Where are my keys?"),
    ] {
        let got = punctuated(&engine, rock);
        assert!(got.iter().any(|v| v == want), "{rock:?} → {got:?}");
    }
}

#[test]
fn punctuate_breaks_run_ons() {
    let Some(engine) = engine() else { return };
    let got = punctuated(&engine, "I was tired I went to bed");
    assert!(got.iter().any(|v| v == "I was tired. I went to bed." || v == "I was tired, I went to bed."), "{got:?}");
}

#[test]
fn punctuate_only_ever_adds_punctuation() {
    let Some(engine) = engine() else { return };
    for rock in ["Honestly I need a break", "I was tired I went to bed", "well i guess thats fine"] {
        for v in punctuated(&engine, rock) {
            assert_eq!(letters(&v), letters(rock), "{v:?}");
        }
    }
}

fn reframes(engine: &Engine, rock: &str) -> Vec<String> {
    engine.variants(rock, 200).into_iter().filter(|v| v.ops.contains(&scribe::Op::Reframe)).map(|v| v.text).collect()
}

/// "I want to X but Y" is often a polite no: the can't reframe turns the
/// "but" clause into the reason ("I can't X because Y").
#[test]
fn cant_reframe_turns_but_into_because() {
    let Some(engine) = engine() else { return };
    let got = reframes(&engine, "I want to come to the party but I have to work late.");
    assert!(got.iter().any(|v| v.contains("can't come to the party because I have to work late")), "{got:?}");
    assert!(!got.iter().any(|v| v.contains("can't come to the party but")), "{got:?}");
    // Other reframes keep "but".
    assert!(got.iter().any(|v| v.contains("need to come to the party but")), "{got:?}");
}

/// Generate (ScribeIE, off by default): never changes ScribeSLM's proposals,
/// their order or text; without an enhancer it adds nothing.
#[test]
fn generate_is_off_by_default_and_inert_without_a_model() {
    let Some(engine) = engine() else { return };
    let rock = "I want to come to the party but I have to work late.";
    let texts = |v: &[scribe::Variant]| -> Vec<String> { v.iter().map(|x| x.text.clone()).collect() };
    let base = engine.variants(rock, 10);
    assert!(base.iter().all(|v| v.enhanced.is_none()), "off by default");
    let on = scribe::scribe_ie::EditOptions { generate: true, ..Default::default() };
    assert!(engine.enhancer.is_none());
    let got = engine.variants_with(rock, 10, &on);
    assert_eq!(texts(&got), texts(&base));
    assert!(got.iter().all(|v| v.enhanced.is_none()), "on without a model adds nothing");
}

/// A stand-in enhancer: ScribeIE's rewrite of each proposal.
struct Celebrate;
impl scribe::scribe_ie::Enhancer for Celebrate {
    fn enhance(&self, _rock: &str, proposal: &str) -> Option<String> {
        proposal.contains("party").then(|| proposal.replace("party", "celebration"))
    }
}

/// A judge that scores every word the same: ScribeIE's words never lose.
struct Indifferent;
impl scribe::engine::SlotScorer for Indifferent {
    fn slot_scores(&self, _: &[String], _: usize, candidates: &[String]) -> Option<Vec<f32>> {
        Some(vec![-1.0; candidates.len()])
    }
}

/// A judge that finds "celebration" unlikely anywhere.
struct NoCelebrations;
impl scribe::engine::SlotScorer for NoCelebrations {
    fn slot_scores(&self, _: &[String], _: usize, candidates: &[String]) -> Option<Vec<f32>> {
        Some(candidates.iter().map(|c| if c == "celebration" { -15.0 } else { -1.0 }).collect())
    }
}

#[test]
fn generate_enhances_every_proposal_without_reordering_them() {
    let Some(mut engine) = engine() else { return };
    let rock = "I want to come to the party but I have to work late.";
    engine.judge = Some(Box::new(Indifferent));
    let base: Vec<String> = engine.variants(rock, 10).into_iter().map(|v| v.text).collect();
    engine.enhancer = Some(Box::new(Celebrate));
    let got = engine.variants_with(rock, 10, &scribe::scribe_ie::EditOptions { generate: true, ..Default::default() });
    assert_eq!(got.iter().map(|v| v.text.clone()).collect::<Vec<_>>(), base, "without ScribeIE ranks, ScribeSLM's order stays");
    for v in &got {
        match &v.enhanced {
            Some(e) => {
                assert_eq!(*e, v.text.replace("party", "celebration"));
                let marked: Vec<&str> = v.enhanced_changed.iter().map(|&(a, b)| &e[a..b]).collect();
                assert_eq!(marked, vec!["celebration"], "marks only what ScribeIE changed");
            }
            None => assert!(!v.text.contains("party")),
        }
    }
    assert!(engine.variants(rock, 10).iter().all(|v| v.enhanced.is_none()), "still off by default");
}

#[test]
fn the_judge_rejects_red_edits_it_finds_less_likely() {
    let Some(mut engine) = engine() else { return };
    let rock = "I want to come to the party but I have to work late.";
    engine.judge = Some(Box::new(NoCelebrations));
    engine.enhancer = Some(Box::new(Celebrate));
    let got = engine.variants_with(rock, 10, &scribe::scribe_ie::EditOptions { generate: true, ..Default::default() });
    assert!(got.iter().any(|v| v.text.contains("party")), "there were proposals to enhance");
    assert!(got.iter().all(|v| v.enhanced.is_none()), "every red edit made the words less likely");
}

/// ScribeIE that prefers proposals with "because" (and writes nothing).
struct PrefersBecause;
impl scribe::scribe_ie::Enhancer for PrefersBecause {
    fn enhance(&self, _: &str, _: &str) -> Option<String> {
        None
    }
    fn rank(&self, _: &str, proposal: &str) -> Option<f32> {
        Some(if proposal.contains("because") { 1.0 } else { 0.0 })
    }
}

#[test]
fn scribe_ie_ranks_order_the_blue_suggestions() {
    let Some(mut engine) = engine() else { return };
    let rock = "I want to come to the party but I have to work late.";
    engine.enhancer = Some(Box::new(PrefersBecause));
    let got = engine.variants_with(rock, 10, &scribe::scribe_ie::EditOptions { generate: true, ..Default::default() });
    let first_other = got.iter().position(|v| !v.text.contains("because")).unwrap_or(got.len());
    assert!(got.iter().any(|v| v.text.contains("because")), "a 'because' proposal exists");
    assert!(got[..first_other].iter().all(|v| v.text.contains("because")) && first_other > 0, "ranked proposals lead");
}

/// Rejects every candidate but the original word.
struct OnlyOriginal;
impl scribe::engine::SlotScorer for OnlyOriginal {
    fn slot_scores(&self, context: &[String], slot: usize, candidates: &[String]) -> Option<Vec<f32>> {
        Some(candidates.iter().map(|c| if *c == context[slot] { 0.0 } else { -100.0 }).collect())
    }
}

/// A judge can replace the usage model's (to compare models on the same rocks).
#[test]
fn a_replacement_judge_decides_which_swaps_survive() {
    let Some(mut engine) = engine() else { return };
    let rock = "The river bank was quiet in the morning.";
    assert!(!swaps(&engine, rock).is_empty(), "the usage model keeps some swaps");
    engine.judge = Some(Box::new(OnlyOriginal));
    assert!(swaps(&engine, rock).is_empty(), "{:?}", swaps(&engine, rock));
}

fn swaps_with(engine: &Engine, rock: &str, options: &scribe::scribe_ie::EditOptions) -> Vec<String> {
    engine.variants_with(rock, 200, options).into_iter().filter(|v| v.ops.contains(&scribe::Op::Swap)).map(|v| v.text).collect()
}

/// The prose-style switch: a request in a style uses that style's judge; a
/// style with no judge loaded (or no style) leaves suggestions as they are.
#[test]
fn a_prose_style_uses_its_own_judge_and_only_its_own() {
    use scribe::engine::ProseStyle;
    use scribe::scribe_ie::EditOptions;
    let Some(mut engine) = engine() else { return };
    engine.style_judges.clear(); // independent of any judges installed in models/judges/
    let rock = "The river bank was quiet in the morning.";
    let plain = swaps(&engine, rock);
    assert!(!plain.is_empty());
    let casual = EditOptions { style: Some(ProseStyle::Casual), ..Default::default() };
    assert_eq!(swaps_with(&engine, rock, &casual), plain, "no casual judge loaded: unchanged");
    engine.style_judges.push((ProseStyle::Casual, Box::new(OnlyOriginal)));
    assert!(swaps_with(&engine, rock, &casual).is_empty(), "the casual judge decides");
    assert_eq!(swaps(&engine, rock), plain, "no style chosen: unchanged");
    let publishable = EditOptions { style: Some(ProseStyle::Publishable), ..Default::default() };
    assert_eq!(swaps_with(&engine, rock, &publishable), plain, "another style's judge is never used");
    assert_eq!(engine.available_styles(), vec![ProseStyle::Casual]);
}

/// The variety slider (0 = fewer, safer swaps; 1 = more, bolder): unset keeps
/// today's suggestions; more variety never offers fewer swaps.
#[test]
fn variety_slider_widens_swaps_monotonically() {
    use scribe::scribe_ie::EditOptions;
    let Some(mut engine) = engine() else { return };
    engine.style_judges.clear();
    let rock = "She gave a brilliant speech about the future of small towns.";
    let count = |v: Option<f32>| swaps_with(&engine, rock, &EditOptions { variety: v, ..Default::default() }).len();
    assert_eq!(swaps_with(&engine, rock, &EditOptions::default()), swaps(&engine, rock), "unset = today");
    let (low, mid, high) = (count(Some(0.0)), count(Some(0.5)), count(Some(1.0)));
    assert!(low <= mid && mid <= high, "{low} {mid} {high}");
    assert!(low < high, "the ends differ: {low} vs {high}");
}
