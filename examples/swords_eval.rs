//! Stages 6–7 (propose and judge word swaps) on SWORDS, the Stanford Word
//! Substitution Benchmark (Lee et al. 2021; CC-BY 3.0): 762 test words in
//! context, each with substitutes rated by human judges.
//!
//!   cargo run --release --example swords_eval [dev|test]
//!
//! Ranking setting (stage 7 alone): every rated substitute, ordered by the
//! usage model's fit in the word's slot; scored by GAP. Generative setting
//! (stages 6 + 7): the engine's own proposals; scored by lenient F@10
//! (substitutes > 50% of judges accepted; outputs the benchmark didn't rate
//! are dropped first). Published (test): humans F@10 48.8 / GAP 66.2;
//! GPT-3 F@10 34.6; BERT F@10 31.7 / GAP 56.9; thesaurus alone F@10 12.0.
//! Data: data/swords/swords-v1.1_{dev,test}.json from github.com/p-lambda/swords.
use scribe::engine::{SlotScorer, CONTENT};
use scribe::lexsub::{gap, pr_at_k};
use scribe::{Engine, Pos};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// The sentence around char `offset` (same rule as text::sentences): its
/// text before the target, and after it.
fn around(context: &[char], offset: usize, len: usize) -> (String, String) {
    let ends = |i: usize| context[i] == '\n' || (matches!(context[i], '.' | '!' | '?') && context.get(i + 1).is_none_or(|n| n.is_whitespace()));
    let start = (0..offset).rev().find(|&i| ends(i)).map_or(0, |i| i + 1);
    let end = (offset + len..context.len()).find(|&i| ends(i)).map_or(context.len(), |i| if context[i] == '\n' { i } else { i + 1 });
    (context[start..offset].iter().collect(), context[offset + len..end].iter().collect())
}

struct Target {
    words: Vec<String>,
    at: usize,
    pos: Option<Pos>,
    /// Rated substitutes (lemmas) with the share of judges who accepted them.
    rated: Vec<(String, f64)>,
}

fn load(split: &str) -> Vec<Target> {
    let d: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("data/swords/swords-v1.1_{split}.json")).unwrap()).unwrap();
    let mut rated: HashMap<&str, Vec<(String, f64)>> = HashMap::new();
    for (sid, sub) in d["substitutes"].as_object().unwrap() {
        let labels: Vec<&str> = d["substitute_labels"][sid].as_array().unwrap().iter().filter_map(|l| l.as_str()).filter(|l| *l != "UNSURE").collect();
        if labels.is_empty() { continue }
        let yes = labels.iter().filter(|l| l.starts_with("TRUE")).count() as f64 / labels.len() as f64;
        rated.entry(sub["target_id"].as_str().unwrap()).or_default().push((sub["substitute"].as_str().unwrap().to_lowercase(), yes));
    }
    let mut out = Vec::new();
    let mut ids: Vec<&String> = d["targets"].as_object().unwrap().keys().collect();
    ids.sort();
    for tid in ids {
        let t = &d["targets"][tid];
        let context: Vec<char> = d["contexts"][t["context_id"].as_str().unwrap()]["context"].as_str().unwrap().chars().collect();
        let word = t["target"].as_str().unwrap();
        let (before, after) = around(&context, t["offset"].as_u64().unwrap() as usize, word.chars().count());
        let mut words = scribe::text::tokenize(&before);
        let at = words.len();
        words.push(word.to_string());
        words.extend(scribe::text::tokenize(&after));
        let pos = match t["pos"].as_str().unwrap() { "NOUN" => Some(Pos::Noun), "VERB" => Some(Pos::Verb), "ADJ" => Some(Pos::Adj), "ADV" => Some(Pos::Adv), _ => None };
        out.push(Target { words, at, pos, rated: rated.remove(tid.as_str()).unwrap_or_default() });
    }
    out
}

/// Candidate lemmas from synsets, in dictionary order, without the word itself.
fn members(e: &Engine, synsets: &[u32], with_similar: bool, lemma: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &s in synsets {
        let syn = &e.lexicon.synsets[s as usize];
        let similar = if with_similar { syn.similar.as_slice() } else { &[] };
        for m in syn.members.iter().chain(similar.iter().flat_map(|&x| e.lexicon.synsets[x as usize].members.iter())) {
            let m = m.to_lowercase();
            if m != lemma && !m.chars().any(|c| c.is_ascii_digit()) && m.split(' ').count() <= 3 && seen.insert(m.clone()) {
                out.push(m);
            }
        }
    }
    out
}

/// Lemmas ordered by the usage model's fit in the slot (inflected like the word), best first, with scores.
fn judged(e: &Engine, words: &[String], at: usize, xpos: &str, upos: &str, lemmas: &[String]) -> Vec<(String, f32)> {
    let Some(scorer) = e.slot_scorer() else { return lemmas.iter().map(|l| (l.clone(), 0.0)).collect() };
    if lemmas.is_empty() { return vec![] }
    let context: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
    let forms: Vec<String> = lemmas.iter().map(|l| if CONTENT.contains(&upos) { e.inflector.inflect(l, xpos) } else { l.clone() }.to_lowercase()).collect();
    let Some(scores) = scorer.slot_scores(&context, at, &forms) else { return vec![] };
    let mut out: Vec<(String, f32)> = lemmas.iter().cloned().zip(scores).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn main() {
    let split = std::env::args().nth(1).unwrap_or_else(|| "test".into());
    let e = Engine::load(Path::new("models")).unwrap();
    let targets = load(&split);
    eprintln!("SWORDS {split}: {} targets", targets.len());

    let (mut gaps, mut gaps_random) = (Vec::new(), Vec::new());
    let systems = ["app (chosen sense, fit margin, ≤3)", "chosen sense, top 10 by fit", "every sense, top 10 by fit", "every sense, dictionary order"];
    let mut generated: Vec<Vec<(HashSet<String>, Vec<String>)>> = vec![Vec::new(); systems.len()];
    let (mut pos_right, mut with_sense) = (0usize, 0usize);
    for (n, t) in targets.iter().enumerate() {
        if n % 100 == 0 { eprintln!("  {n}/{}", targets.len()) }
        let parsed = e.parse_tokens(&t.words);
        let tok = &parsed.tokens[t.at];
        let lemma = tok.lemma.to_lowercase();
        pos_right += (Pos::from_upos(&tok.upos) == t.pos) as usize;
        let rated: Vec<(String, f64)> = t.rated.iter().filter(|(s, _)| *s != lemma).cloned().collect();
        let names: HashSet<String> = rated.iter().map(|(s, _)| s.clone()).collect();
        let accepted: HashSet<String> = rated.iter().filter(|(_, w)| *w > 0.5).map(|(s, _)| s.clone()).collect();

        // Ranking: every rated substitute, by fit.
        let all: Vec<String> = rated.iter().map(|(s, _)| s.clone()).collect();
        let order: Vec<String> = judged(&e, &t.words, t.at, &tok.xpos, &tok.upos, &all).into_iter().map(|(s, _)| s).collect();
        gaps.push(gap(&rated, &order));
        gaps_random.push(gap(&rated, &all)); // benchmark file order (ids are hashes)

        // Generation: the engine's own candidates.
        let adj = tok.upos == "ADJ";
        let chosen: Vec<u32> = tok.sense.filter(|_| tok.sense_supported).into_iter().collect();
        with_sense += (!chosen.is_empty()) as usize;
        let every: Vec<u32> = Pos::from_upos(&tok.upos).map(|p| e.lexicon.senses(&lemma, p).to_vec()).unwrap_or_default();
        let lenient = |list: Vec<String>| -> Vec<String> { list.into_iter().filter(|s| names.contains(s)).collect() };
        let original = judged(&e, &t.words, t.at, &tok.xpos, &tok.upos, std::slice::from_ref(&lemma)).first().map_or(0.0, |x| x.1);
        let chosen_fit = judged(&e, &t.words, t.at, &tok.xpos, &tok.upos, &members(&e, &chosen, adj, &lemma));
        let app = scribe::variants::keep_fitting(original, chosen_fit.clone(), scribe::variants::SWAP_MARGIN, scribe::variants::SWAPS_PER_WORD);
        let lists = [
            app,
            chosen_fit.into_iter().map(|x| x.0).collect(),
            judged(&e, &t.words, t.at, &tok.xpos, &tok.upos, &members(&e, &every, adj, &lemma)).into_iter().map(|x| x.0).collect(),
            members(&e, &every, adj, &lemma),
        ];
        for (g, list) in generated.iter_mut().zip(lists) {
            g.push((accepted.clone(), lenient(list)));
        }
    }
    let mean = |v: &[f64]| 100.0 * v.iter().sum::<f64>() / v.len().max(1) as f64;
    let n = targets.len() as f64;
    println!("SWORDS {split} ({} targets): our part of speech matches the benchmark's for {:.1}%; a meaning with options chosen for {:.1}%",
             targets.len(), 100.0 * pos_right as f64 / n, 100.0 * with_sense as f64 / n);
    println!("ranking (stage 7): GAP {:.1}  (unordered candidate list {:.1})", mean(&gaps), mean(&gaps_random));
    println!("generation (stages 6+7), lenient:");
    for (name, g) in systems.iter().zip(&generated) {
        let (p, r, f) = pr_at_k(g, 10);
        println!("  {name:<36} F@10 {f:.1}  (P {p:.1}, R {r:.1})");
    }
}
