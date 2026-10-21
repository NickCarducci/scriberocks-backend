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
//! LM=bert (BERT_DIR=models/bert-mini-usage[-dictionary] for the adapted ones) or USAGE=<models dir> swaps the language model; POS=ADJ|NOUN|VERB|ADV
//! and LIMIT=n score a subset (quick pilots). PROPOSER=bert: BERT-mini proposes, the LM above judges.
//!
//! The judge combines fit (the usage model's log-probability of the
//! candidate in the masked slot) with closeness to the original word, as
//! BERT-based substitution does (Zhou et al. 2019): cosine between the
//! original and the substituted sentence's vectors, word visible, at the slot
//! and averaged over the other words. Weights are tuned on dev.
use scribe::engine::CONTENT;
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
        if std::env::var("POS").is_ok_and(|want| t["pos"].as_str() != Some(want.as_str())) { continue }
        out.push(Target { words, at, pos, rated: rated.remove(tid.as_str()).unwrap_or_default() });
    }
    // LIMIT=n: the first n targets (ids are hashes, so effectively a random sample).
    if let Some(n) = std::env::var("LIMIT").ok().and_then(|v| v.parse().ok()) { out.truncate(n) }
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

/// The language model under test: the shipped usage model by default;
/// USAGE=<models/ dir name> for another usage model (pilots); LM=bert for
/// BERT-mini (data/bert-mini). The engine's own tagging and meanings stay
/// as shipped either way.
trait Lm: Send + Sync {
    fn fit(&self, context: &[String], at: usize, forms: &[String]) -> Option<Vec<f32>>;
    fn vectors(&self, sentences: &[Vec<String>]) -> Option<Vec<Vec<Vec<f32>>>>;
    fn top(&self, words: &[String], at: usize, k: usize, visible: bool) -> Vec<(String, f32)>;
    fn unknown(&self, word: &str) -> bool;
}

impl Lm for scribe::usage::model::UsageModel {
    fn fit(&self, context: &[String], at: usize, forms: &[String]) -> Option<Vec<f32>> { self.candidate_scores(context, at, forms).ok() }
    fn vectors(&self, sentences: &[Vec<String>]) -> Option<Vec<Vec<Vec<f32>>>> { self.contextual_batch(sentences).ok() }
    fn top(&self, words: &[String], at: usize, k: usize, visible: bool) -> Vec<(String, f32)> { self.predict(words, at, k, visible).unwrap_or_default() }
    fn unknown(&self, word: &str) -> bool { self.encode_words(&word.split(' ').map(String::from).collect::<Vec<_>>()).0.contains(&scribe::usage::vocab::UNK) }
}

impl Lm for scribe::bert::Bert {
    fn fit(&self, context: &[String], at: usize, forms: &[String]) -> Option<Vec<f32>> { self.slot_scores(context, at, forms).ok() }
    fn vectors(&self, sentences: &[Vec<String>]) -> Option<Vec<Vec<Vec<f32>>>> { self.contextual_batch(sentences).ok() }
    fn top(&self, words: &[String], at: usize, k: usize, visible: bool) -> Vec<(String, f32)> { self.predict(words, at, k, visible).unwrap_or_default() }
    fn unknown(&self, word: &str) -> bool { word.split(' ').any(|w| self.word_pieces(w).iter().any(|p| p == "[UNK]")) }
}

static LM: std::sync::OnceLock<Box<dyn Lm>> = std::sync::OnceLock::new();
/// The judge (and the default proposer).
fn lm() -> &'static dyn Lm { LM.get().expect("language model loaded").as_ref() }
/// PROPOSER=bert (BERT-mini, raw) proposes while LM/USAGE judges: a broad
/// casual source narrowed by the standards judge.
static PROPOSER: std::sync::OnceLock<Box<dyn Lm>> = std::sync::OnceLock::new();
fn proposer() -> &'static dyn Lm { PROPOSER.get().map(|b| b.as_ref()).unwrap_or_else(lm) }

/// Lemmas ordered by the usage model's fit in the slot (inflected like the word), best first, with scores.
fn judged(e: &Engine, words: &[String], at: usize, xpos: &str, upos: &str, lemmas: &[String]) -> Vec<(String, f32)> {
    let scorer = lm();
    if lemmas.is_empty() { return vec![] }
    let context: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
    let forms: Vec<String> = lemmas.iter().map(|l| if CONTENT.contains(&upos) { e.inflector.inflect(l, xpos) } else { l.clone() }.to_lowercase()).collect();
    let Some(scores) = scorer.fit(&context, at, &forms) else { return vec![] };
    let mut out: Vec<(String, f32)> = lemmas.iter().cloned().zip(scores).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    dot / (norm(a) * norm(b))
}

/// Per candidate: (fit, closeness at the slot, closeness over the rest of the sentence).
fn components(e: &Engine, words: &[String], at: usize, xpos: &str, upos: &str, lemmas: &[String]) -> Vec<(f32, f32, f32)> {
    let (u, scorer) = (lm(), lm());
    if lemmas.is_empty() { return vec![] }
    let lower: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
    let forms: Vec<String> = lemmas.iter().map(|l| if CONTENT.contains(&upos) { e.inflector.inflect(l, xpos) } else { l.clone() }.to_lowercase()).collect();
    let fit = scorer.fit(&lower, at, &forms).unwrap_or_else(|| vec![0.0; forms.len()]);
    let mut sentences = vec![lower.clone()];
    for f in &forms {
        let mut s = lower.clone();
        s.splice(at..at + 1, f.split(' ').map(String::from));
        sentences.push(s);
    }
    let Some(vectors) = u.vectors(&sentences) else { return fit.into_iter().map(|f| (f, 0.0, 0.0)).collect() };
    let original = &vectors[0];
    forms.iter().enumerate().map(|(k, f)| {
        let v = &vectors[k + 1];
        let extra = f.split(' ').count() - 1;
        let slot = match (original.get(at), v.get(at)) { (Some(a), Some(b)) => cosine(a, b), _ => 0.0 };
        let others: Vec<f32> = (0..original.len()).filter(|&j| j != at).filter_map(|j| {
            let j2 = if j > at { j + extra } else { j };
            v.get(j2).map(|b| cosine(&original[j], b))
        }).collect();
        let rest = if others.is_empty() { 0.0 } else { others.iter().sum::<f32>() / others.len() as f32 };
        (fit[k], slot, rest)
    }).collect()
}

/// One target's rated candidates with their judge components, after our parse.
struct Scored {
    rated: Vec<(String, f64)>,
    parts: Vec<(f32, f32, f32)>,
}

fn rank(sc: &Scored, w: (f32, f32)) -> Vec<String> {
    let mut idx: Vec<usize> = (0..sc.rated.len()).collect();
    let total = |i: usize| { let (f, a, b) = sc.parts[i]; f + w.0 * a + w.1 * b };
    idx.sort_by(|&a, &b| total(b).total_cmp(&total(a)));
    idx.into_iter().map(|i| sc.rated[i].0.clone()).collect()
}

fn ranking_set(e: &Engine, targets: &[Target]) -> Vec<Scored> {
    targets.iter().map(|t| {
        let parsed = e.parse_tokens(&t.words);
        let tok = &parsed.tokens[t.at];
        let lemma = tok.lemma.to_lowercase();
        let rated: Vec<(String, f64)> = t.rated.iter().filter(|(s, _)| *s != lemma).cloned().collect();
        let lemmas: Vec<String> = rated.iter().map(|(s, _)| s.clone()).collect();
        let parts = components(e, &t.words, t.at, &tok.xpos, &tok.upos, &lemmas);
        Scored { rated, parts }
    }).collect()
}

/// Synsets related to `synsets`: (more general, more specific, sisters).
fn related(e: &Engine, hyponyms: &HashMap<u32, Vec<u32>>, synsets: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let up: Vec<u32> = synsets.iter().flat_map(|&s| e.lexicon.synsets[s as usize].hypernyms.iter().copied()).collect();
    let down: Vec<u32> = synsets.iter().flat_map(|s| hyponyms.get(s).into_iter().flatten().copied()).collect();
    let sisters: Vec<u32> = up.iter().flat_map(|s| hyponyms.get(s).into_iter().flatten().copied()).filter(|s| !synsets.contains(s)).collect();
    (up, down, sisters)
}

/// Lists longer than this are cut (dictionary order) before judging.
const MAX_PROPOSALS: usize = 200;

const WEIGHTS: &[f32] = &[0.0, 5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0];

fn main() {
    let split = std::env::args().nth(1).unwrap_or_else(|| "test".into());
    let e = Engine::load(Path::new("models")).unwrap();
    let model: Box<dyn Lm> = match (std::env::var("LM").as_deref(), std::env::var("USAGE")) {
        (Ok("bert"), _) => {
            let dir = std::env::var("BERT_DIR").unwrap_or_else(|_| "data/bert-mini".into());
            eprintln!("language model: BERT-format model in {dir}");
            Box::new(scribe::bert::Bert::load(Path::new(&dir), scribe::usage::model::Device::Cpu).unwrap())
        }
        (_, Ok(dir)) => Box::new(scribe::usage::model::UsageModel::load(&Path::new("models").join(dir), scribe::usage::model::Device::Cpu).unwrap()),
        _ => Box::new(scribe::usage::model::UsageModel::load(Path::new("models/usage"), scribe::usage::model::Device::Cpu).unwrap()),
    };
    let _ = LM.set(model);
    if std::env::var("PROPOSER").as_deref() == Ok("bert") {
        eprintln!("proposer: BERT-mini (data/bert-mini)");
        let _ = PROPOSER.set(Box::new(scribe::bert::Bert::load(Path::new("data/bert-mini"), scribe::usage::model::Device::Cpu).unwrap()));
    }
    let targets = load(&split);
    eprintln!("SWORDS {split}: {} targets", targets.len());

    // Stage 7: tune the closeness weights on dev, score on this split.
    eprintln!("judge components: dev, then {split}");
    let dev = ranking_set(&e, &load("dev"));
    let here = ranking_set(&e, &targets);
    let mean_gap = |set: &[Scored], w: (f32, f32)| 100.0 * set.iter().map(|sc| gap(&sc.rated, &rank(sc, w))).sum::<f64>() / set.len().max(1) as f64;
    let grid: Vec<(f32, f32)> = WEIGHTS.iter().flat_map(|&a| WEIGHTS.iter().map(move |&b| (a, b))).collect();
    let best = |fit_weight_on: bool| -> (f32, f32) {
        // Without fit, scale closeness up so fit (log-prob) is negligible.
        let options: Vec<(f32, f32)> = if fit_weight_on { grid.clone() } else { vec![(1e4, 0.0), (0.0, 1e4), (1e4, 1e4)] };
        options.into_iter().max_by(|&a, &b| mean_gap(&dev, a).total_cmp(&mean_gap(&dev, b))).unwrap()
    };
    let (combined, closeness) = (best(true), best(false));
    // Candidates the usage model can't spell from known pieces (judged blind).
    {
        let u = lm();
        let all: Vec<&String> = targets.iter().flat_map(|t| t.rated.iter().map(|(s, _)| s)).collect();
        let unknown = all.iter().filter(|s| u.unknown(s)).count();
        println!("candidates with a piece unknown to the usage model: {:.1}% of {}", 100.0 * unknown as f64 / all.len() as f64, all.len());
    }
    println!("ranking (stage 7), GAP on {split}:");
    println!("  fit alone (shipped judge)            {:.1}", mean_gap(&here, (0.0, 0.0)));
    println!("  closeness alone {:<20} {:.1}", format!("{closeness:?}"), mean_gap(&here, closeness));
    println!("  fit + closeness {:<20} {:.1}  (weights tuned on dev: {:.1})", format!("{combined:?}"), mean_gap(&here, combined), mean_gap(&dev, combined));

    let (mut gaps, mut gaps_random) = (Vec::new(), Vec::new());
    let systems = ["app (chosen sense, fit margin, ≤3)", "chosen sense, top 10 by fit", "every sense, top 10 by fit", "every sense, dictionary order",
                   "chosen sense, top 3 by new judge", "chosen sense, top 10 by new judge", "every sense, top 10 by new judge"];
    let mut generated: Vec<Vec<(HashSet<String>, Vec<String>)>> = vec![Vec::new(); systems.len()];
    let mut hyponyms: HashMap<u32, Vec<u32>> = HashMap::new();
    for (i, syn) in e.lexicon.synsets.iter().enumerate() {
        for &h in &syn.hypernyms {
            hyponyms.entry(h).or_default().push(i as u32);
        }
    }
    let sources = ["chosen meaning: synonyms", "  + more general", "  + more specific", "  + sisters", "every meaning: synonyms", "every meaning: all four",
                   "model, word hidden (top 50)", "model, word visible (top 50)", "model, both combined (top 50)", "dictionary (chosen+3) + model (top 20)"];
    // Per source: (perfect-judge top 10, new-judge top 10), and list sizes.
    // Per source: perfect judge, new judge, and the proposer's own order.
    type Lists = Vec<(HashSet<String>, Vec<String>)>;
    let mut widened: Vec<(Lists, Lists, Lists, Lists)> = vec![(Vec::new(), Vec::new(), Vec::new(), Vec::new()); sources.len()];
    let mut sizes = vec![0usize; sources.len()];
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
        let judge = |lemmas: &[String], k: usize| -> Vec<String> {
            let parts = components(&e, &t.words, t.at, &tok.xpos, &tok.upos, lemmas);
            let sc = Scored { rated: lemmas.iter().map(|l| (l.clone(), 0.0)).collect(), parts };
            rank(&sc, combined).into_iter().take(k).collect()
        };
        let lenient = |list: Vec<String>| -> Vec<String> { list.into_iter().filter(|s| names.contains(s)).collect() };
        let original = judged(&e, &t.words, t.at, &tok.xpos, &tok.upos, std::slice::from_ref(&lemma)).first().map_or(0.0, |x| x.1);
        let chosen_fit = judged(&e, &t.words, t.at, &tok.xpos, &tok.upos, &members(&e, &chosen, adj, &lemma));
        let app = scribe::variants::keep_fitting(original, chosen_fit.clone(), scribe::variants::SWAP_MARGIN, scribe::variants::SWAPS_PER_WORD);
        let lists = [
            app,
            chosen_fit.into_iter().map(|x| x.0).collect(),
            judged(&e, &t.words, t.at, &tok.xpos, &tok.upos, &members(&e, &every, adj, &lemma)).into_iter().map(|x| x.0).collect(),
            members(&e, &every, adj, &lemma),
            judge(&members(&e, &chosen, adj, &lemma), 3),
            judge(&members(&e, &chosen, adj, &lemma), 10),
            judge(&members(&e, &every, adj, &lemma), 10),
        ];
        for (g, list) in generated.iter_mut().zip(lists) {
            g.push((accepted.clone(), lenient(list)));
        }

        // Wider proposals: the perfect-judge ceiling and the new judge.
        let (up, down, sis) = related(&e, &hyponyms, &chosen);
        let (eup, edown, esis) = related(&e, &hyponyms, &every);
        let cat = |parts: &[&[u32]]| -> Vec<u32> { parts.concat() };
        // Generated: the usage model's own top words for the slot, as lemmas
        // usable in the word's part of speech.
        let generated_by = |mode: u8, k: usize| -> Vec<String> {
            let (u, Some(p)) = (proposer(), Pos::from_upos(&tok.upos)) else { return vec![] };
            let lower: Vec<String> = t.words.iter().map(|w| w.to_lowercase()).collect();
            let all = |visible: bool| -> HashMap<String, f32> { u.top(&lower, t.at, usize::MAX, visible).into_iter().collect() };
            let mut scored: Vec<(String, f32)> = match mode {
                0 => all(false).into_iter().collect(),
                1 => all(true).into_iter().collect(),
                _ => { let v = all(true); all(false).into_iter().map(|(w, s)| { let x = v.get(&w).copied().unwrap_or(-30.0); (w, s + x) }).collect() }
            };
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            let mut seen = HashSet::new();
            let mut out = Vec::new();
            for (form, _) in scored {
                if out.len() == k { break }
                let Some(l) = e.lexicon.lemmatize(&form, Some(p)).into_iter().next() else { continue };
                let l = l.to_lowercase();
                if l != lemma && !e.lexicon.senses(&l, p).is_empty() && seen.insert(l.clone()) { out.push(l) }
            }
            out
        };
        let dictionary = members(&e, &cat(&[&chosen, &up, &down]), adj, &lemma);
        let mut hybrid = dictionary.clone();
        hybrid.truncate(MAX_PROPOSALS);
        for w in generated_by(2, 20) { if !hybrid.contains(&w) { hybrid.push(w) } }
        let lists: [Vec<String>; 10] = [
            members(&e, &chosen, adj, &lemma),
            members(&e, &cat(&[&chosen, &up]), adj, &lemma),
            members(&e, &cat(&[&chosen, &up, &down]), adj, &lemma),
            members(&e, &cat(&[&chosen, &up, &down, &sis]), adj, &lemma),
            members(&e, &every, adj, &lemma),
            members(&e, &cat(&[&every, &eup, &edown, &esis]), adj, &lemma),
            generated_by(0, 50),
            generated_by(1, 50),
            generated_by(2, 50),
            hybrid,
        ];
        let weight: HashMap<&str, f64> = rated.iter().map(|(s, w)| (s.as_str(), *w)).collect();
        for (k, mut list) in lists.into_iter().enumerate() {
            list.truncate(MAX_PROPOSALS);
            sizes[k] += list.len();
            let mut perfect = list.clone();
            perfect.sort_by(|a, b| weight.get(b.as_str()).unwrap_or(&-1.0).total_cmp(weight.get(a.as_str()).unwrap_or(&-1.0)));
            widened[k].0.push((accepted.clone(), lenient(perfect.into_iter().take(10).collect())));
            widened[k].1.push((accepted.clone(), lenient(judge(&list, 10))));
            widened[k].2.push((accepted.clone(), lenient(list.iter().take(10).cloned().collect())));
            widened[k].3.push((accepted.clone(), list.iter().take(10).cloned().collect()));
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
    println!("wider proposals (lenient F@10): average list size, perfect judge, new judge:");
    for (k, name) in sources.iter().enumerate() {
        let (_, _, ceiling) = pr_at_k(&widened[k].0, 10);
        let (p, r, f) = pr_at_k(&widened[k].1, 10);
        let (rp, rr, rf) = pr_at_k(&widened[k].2, 10);
        println!("  {name:<38} {:>6.1} words   ceiling {ceiling:.1}   judged {f:.1} (P {p:.1}, R {r:.1})   as proposed {rf:.1} (P {rp:.1}, R {rr:.1}), strict {:.1}", sizes[k] as f64 / n, pr_at_k(&widened[k].3, 10).2);
    }
}
