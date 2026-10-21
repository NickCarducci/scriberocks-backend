//! ScribeSLM on real writers' revisions: IteraTeR (Grammarly research, Du et
//! al., ACL 2022; Apache 2.0) — sentence pairs before/after a human revision,
//! labelled by intent (clarity, fluency, coherence, style, meaning-changed).
//! Test data only; never in any training corpus.
//!
//!   cargo run --release --example iterater_eval [human|full] [limit]
//!   LM=bert [BERT_DIR=...] or USAGE=<models dir> picks the language model.
//!
//! 1. Judging: does the model prefer the writer's revision over the original?
//!    Score = mean masked log-probability per word (each word hidden in turn).
//!    Chance 50%. Meaning-changed edits are left out (no judge can know the
//!    writer's intended meaning). Shown beside the trigram fluency model (live)
//!    and "the shorter sentence wins".
//! 2. Proposing: revisions that change exactly one word. Is the writer's new
//!    word among the dictionary's synonyms of the chosen meaning, of any
//!    meaning, or the model's own top 10 / 50 for the slot (word visible)?
use scribe::{text, Engine, Pos};
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

trait Lm {
    /// Log-probability of `words[at]` filling its own slot (hidden).
    fn word_score(&self, words: &[String], at: usize) -> f32;
    fn top(&self, words: &[String], at: usize, k: usize) -> Vec<String>;
}

impl Lm for scribe::usage::model::UsageModel {
    fn word_score(&self, words: &[String], at: usize) -> f32 {
        self.candidate_scores(words, at, std::slice::from_ref(&words[at])).ok().and_then(|v| v.first().copied()).unwrap_or(-30.0)
    }
    fn top(&self, words: &[String], at: usize, k: usize) -> Vec<String> {
        self.predict(words, at, k, true).unwrap_or_default().into_iter().map(|x| x.0).collect()
    }
}

impl Lm for scribe::bert::Bert {
    fn word_score(&self, words: &[String], at: usize) -> f32 {
        self.slot_scores(words, at, std::slice::from_ref(&words[at])).ok().and_then(|v| v.first().copied()).unwrap_or(-30.0)
    }
    fn top(&self, words: &[String], at: usize, k: usize) -> Vec<String> {
        self.predict(words, at, k, true).unwrap_or_default().into_iter().map(|x| x.0).collect()
    }
}

/// Mean log-probability per word, each word hidden in turn (pseudo-log-likelihood).
fn sentence_score(lm: &dyn Lm, words: &[String]) -> f32 {
    if words.is_empty() { return 0.0 }
    (0..words.len()).map(|i| lm.word_score(words, i)).sum::<f32>() / words.len() as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let set = std::env::args().nth(1).unwrap_or_else(|| "human".into());
    let limit: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let e = Engine::load(Path::new("models"))?;
    let cpu = scribe::usage::model::Device::Cpu;
    let (lm, name): (Box<dyn Lm>, String) = match (std::env::var("LM").as_deref(), std::env::var("USAGE")) {
        (Ok("bert"), _) => {
            let dir = std::env::var("BERT_DIR").unwrap_or_else(|_| "data/bert-mini".into());
            (Box::new(scribe::bert::Bert::load(Path::new(&dir), cpu)?), format!("BERT-format model in {dir}"))
        }
        (_, Ok(dir)) => (Box::new(scribe::usage::model::UsageModel::load(&Path::new("models").join(&dir), cpu)?), format!("usage model models/{dir}")),
        _ => (Box::new(scribe::usage::model::UsageModel::load(Path::new("models/usage"), cpu)?), "from-scratch usage model (live)".into()),
    };
    let path = format!("data/iterater/IteraTeR/{set}_sent_level/test.json");
    let pairs: Vec<serde_json::Value> = std::fs::read_to_string(&path)?.lines().filter_map(|l| serde_json::from_str(l).ok()).take(limit).collect();
    let lower = |s: &str| -> Vec<String> { text::tokenize(s).into_iter().map(|w| w.to_lowercase()).collect() };
    println!("IteraTeR {set} test: {} revisions; language model: {name}", pairs.len());

    // 1. Judging, by intent: (n, model right, trigram right, shorter right).
    let mut by: BTreeMap<String, (usize, usize, usize, usize)> = BTreeMap::new();
    for (k, p) in pairs.iter().enumerate() {
        if k % 100 == 0 { eprintln!("  judging {k}/{}", pairs.len()) }
        let intent = p["labels"].as_str().unwrap_or("?").to_string();
        if intent == "meaning-changed" || intent == "others" { continue }
        let (b, a) = (lower(p["before_sent"].as_str().unwrap_or("")), lower(p["after_sent"].as_str().unwrap_or("")));
        if b.is_empty() || a.is_empty() || b == a { continue }
        let model = sentence_score(lm.as_ref(), &a) > sentence_score(lm.as_ref(), &b);
        let trigram = e.lm.score(&a) > e.lm.score(&b);
        let shorter = a.len() < b.len();
        for key in [intent.clone(), "all (no meaning-changed)".to_string()] {
            let r = by.entry(key).or_default();
            r.0 += 1; r.1 += model as usize; r.2 += trigram as usize; r.3 += shorter as usize;
        }
    }
    println!("judging: prefers the writer's revision (chance 50%)");
    println!("  {:<26} {:>5}  {:>8}  {:>13}  {:>8}", "intent", "n", "model", "trigram (live)", "shorter");
    for (intent, (n, m, t, s)) in &by {
        let pct = |x: usize| 100.0 * x as f64 / *n as f64;
        println!("  {intent:<26} {n:>5}  {:>7.1}%  {:>13.1}%  {:>7.1}%", pct(*m), pct(*t), pct(*s));
    }

    // 2. Proposing: one-word swaps.
    let (mut n, mut chosen, mut every, mut top10, mut top50) = (0, 0, 0, 0, 0);
    for p in &pairs {
        let (b, a) = (text::tokenize(p["before_sent"].as_str().unwrap_or("")), text::tokenize(p["after_sent"].as_str().unwrap_or("")));
        if b.len() != a.len() { continue }
        let diff: Vec<usize> = (0..b.len()).filter(|&i| !b[i].eq_ignore_ascii_case(&a[i])).collect();
        if diff.len() != 1 { continue }
        let i = diff[0];
        if !b[i].chars().all(char::is_alphabetic) || !a[i].chars().all(char::is_alphabetic) { continue }
        let parsed = e.parse_tokens(&b);
        let tok = &parsed.tokens[i];
        let Some(pos) = Pos::from_upos(&tok.upos) else { continue };
        let lemma_of = |w: &str| e.lexicon.lemmatize(w, Some(pos)).into_iter().next().unwrap_or_else(|| w.to_lowercase()).to_lowercase();
        let want = lemma_of(&a[i]);
        n += 1;
        let members = |senses: &[u32]| -> HashSet<String> { senses.iter().flat_map(|&s| e.lexicon.synsets[s as usize].members.iter().map(|m| m.to_lowercase())).collect() };
        chosen += tok.sense.is_some_and(|s| members(&[s]).contains(&want)) as usize;
        every += members(e.lexicon.senses(&tok.lemma.to_lowercase(), pos)).contains(&want) as usize;
        let lowered: Vec<String> = b.iter().map(|w| w.to_lowercase()).collect();
        let top: Vec<String> = lm.top(&lowered, i, 50).into_iter().map(|w| lemma_of(&w)).collect();
        top10 += top.iter().take(10).any(|w| *w == want) as usize;
        top50 += top.contains(&want) as usize;
    }
    let pct = |x: usize| 100.0 * x as f64 / n.max(1) as f64;
    println!("proposing: {n} one-word swaps by writers; the writer's word is among");
    println!("  synonyms of the chosen meaning (live)   {:.1}%", pct(chosen));
    println!("  synonyms of any meaning                 {:.1}%", pct(every));
    println!("  the model's top 10 / top 50             {:.1}% / {:.1}%", pct(top10), pct(top50));
    Ok(())
}
