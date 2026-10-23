//! ProLex (Zhang et al., Findings of ACL 2024; Apache 2.0 repository): word
//! substitution in learners' own essays (TOEFL-11), human-judged for meaning,
//! collocation, grammar and proficiency. 680 test items, each a target word in
//! a sentence with substitutes judged acceptable or not. Test data only; the
//! TOEFL-11 sentences are licensed by ETS, so never in training or shipped.
//!
//!   cargo run --release --example prolex_eval [usage | <BERT-format dir>]...   (PIECE_MEAN=1: average piece scores)
//!
//! Judging: rank each item's acceptable and unacceptable substitutes by the
//! model's fit in the slot (inflected like the target); GAP with acceptable = 1.
//! Proposing: F@10 (lemmas) against the acceptable substitutes, for the
//! dictionary's synonyms (chosen meaning / any meaning) and each model's own
//! top 10 for the slot (word visible). Published: GPT-4 zero-shot F@10 50.5.
use scribe::engine::{SlotScorer, CONTENT};
use scribe::lexsub::{gap, pr_at_k};
use scribe::{text, Engine, Pos};
use std::collections::HashSet;
use std::path::Path;

trait Lm {
    fn fit(&self, words: &[String], at: usize, forms: &[String]) -> Option<Vec<f32>>;
    fn top(&self, words: &[String], at: usize, k: usize) -> Vec<String>;
}
impl Lm for scribe::usage::model::UsageModel {
    fn fit(&self, w: &[String], at: usize, f: &[String]) -> Option<Vec<f32>> { self.slot_scores(w, at, f) }
    fn top(&self, w: &[String], at: usize, k: usize) -> Vec<String> { self.predict(w, at, k, true).unwrap_or_default().into_iter().map(|x| x.0).collect() }
}
impl Lm for scribe::bert::Bert {
    /// PIECE_MEAN=1: average a candidate's piece scores instead of summing them
    /// (a sum penalizes candidates that split into several pieces).
    fn fit(&self, w: &[String], at: usize, f: &[String]) -> Option<Vec<f32>> {
        let s = self.slot_scores(w, at, f).ok()?;
        if std::env::var_os("PIECE_MEAN").is_none() { return Some(s) }
        Some(s.iter().zip(f).map(|(x, c)| x / c.split(' ').map(|p| self.word_pieces(p).len()).sum::<usize>().max(1) as f32).collect())
    }
    fn top(&self, w: &[String], at: usize, k: usize) -> Vec<String> { self.predict(w, at, k, true).unwrap_or_default().into_iter().map(|x| x.0).collect() }
}

struct Item {
    words: Vec<String>,
    at: usize,
    acceptable: Vec<String>,
    unacceptable: Vec<String>,
}

fn list(cell: &str) -> Vec<String> {
    cell.trim_matches(|c| c == '[' || c == ']').split(',').map(|s| s.trim().trim_matches(|c| c == '\'' || c == '"').to_lowercase()).filter(|s| !s.is_empty()).collect()
}

/// Minimal CSV reader (quoted fields with doubled quotes).
fn csv_rows(text: &str) -> Vec<Vec<String>> {
    let (mut rows, mut row, mut field, mut quoted) = (Vec::new(), Vec::new(), String::new(), false);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => { field.push('"'); chars.next(); }
            ('"', _) => quoted = !quoted,
            (',', false) => row.push(std::mem::take(&mut field)),
            ('\n', false) => { row.push(std::mem::take(&mut field)); rows.push(std::mem::take(&mut row)); }
            ('\r', false) => {}
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() { row.push(field); rows.push(row) }
    rows
}

fn load() -> Vec<Item> {
    let rows = csv_rows(&std::fs::read_to_string("data/prolex/ProLex_v1.0_test.csv").unwrap());
    rows.iter().skip(1).filter(|r| r.len() >= 4).filter_map(|r| {
        let marked = &r[1];
        let start = marked.find("**")?;
        let end = start + 2 + marked[start + 2..].find("**")?;
        let mut words = text::tokenize(&marked[..start]);
        let at = words.len();
        words.push(marked[start + 2..end].to_string());
        words.extend(text::tokenize(&marked[end + 2..]));
        Some(Item { words, at, acceptable: list(&r[2]), unacceptable: list(&r[3]) })
    }).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let e = Engine::load(Path::new("models"))?;
    let cpu = scribe::usage::model::Device::Cpu;
    let names: Vec<String> = std::env::args().skip(1).collect();
    let models: Vec<(String, Box<dyn Lm>)> = names.iter().map(|n| -> Result<(String, Box<dyn Lm>), Box<dyn std::error::Error>> {
        Ok((n.clone(), if n == "usage" { Box::new(scribe::usage::model::UsageModel::load(Path::new("models/usage"), cpu.clone())?) } else { Box::new(scribe::bert::Bert::load(Path::new(n), cpu.clone())?) }))
    }).collect::<Result<_, _>>()?;
    let items = load();
    println!("ProLex test: {} items (learners' essays, human-judged substitutes)", items.len());

    let (mut chosen, mut every) = (Vec::new(), Vec::new());
    let mut gaps = vec![Vec::new(); models.len()];
    let mut own = vec![Vec::new(); models.len()];
    for it in &items {
        let parsed = e.parse_tokens(&it.words);
        let tok = &parsed.tokens[it.at];
        let Some(pos) = Pos::from_upos(&tok.upos) else { continue };
        let lemma_of = |w: &str| e.lexicon.lemmatize(w, Some(pos)).into_iter().next().unwrap_or_else(|| w.to_lowercase()).to_lowercase();
        let accepted: HashSet<String> = it.acceptable.iter().map(|w| lemma_of(w)).collect();
        let target = tok.lemma.to_lowercase();
        let members = |senses: &[u32]| -> Vec<String> {
            let mut seen = HashSet::new();
            senses.iter().flat_map(|&s| e.lexicon.synsets[s as usize].members.iter().map(|m| m.to_lowercase())).filter(|m| *m != target && seen.insert(m.clone())).collect()
        };
        chosen.push((accepted.clone(), members(&tok.sense.into_iter().collect::<Vec<_>>()).into_iter().take(10).collect()));
        every.push((accepted.clone(), members(e.lexicon.senses(&target, pos)).into_iter().take(10).collect()));
        let lower: Vec<String> = it.words.iter().map(|w| w.to_lowercase()).collect();
        let rated: Vec<(String, f64)> = it.acceptable.iter().map(|w| (w.clone(), 1.0)).chain(it.unacceptable.iter().map(|w| (w.clone(), 0.0))).collect();
        let forms: Vec<String> = rated.iter().map(|(w, _)| if CONTENT.contains(&tok.upos.as_str()) && !w.contains(' ') { e.inflector.inflect(w, &tok.xpos) } else { w.clone() }.to_lowercase()).collect();
        for (k, (_, m)) in models.iter().enumerate() {
            if let Some(s) = m.fit(&lower, it.at, &forms) {
                let mut idx: Vec<usize> = (0..rated.len()).collect();
                idx.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
                gaps[k].push(gap(&rated, &idx.iter().map(|&i| rated[i].0.clone()).collect::<Vec<_>>()));
            }
            let mut seen = HashSet::new();
            let top: Vec<String> = m.top(&lower, it.at, 50).into_iter().map(|w| lemma_of(&w)).filter(|w| *w != target && seen.insert(w.clone())).take(10).collect();
            own[k].push((accepted.clone(), top));
        }
    }
    let f = |v: &[(HashSet<String>, Vec<String>)]| pr_at_k(v, 10);
    println!("proposing, F@10 (P, R) against human-accepted substitutes (lemmas):");
    let (p, r, x) = f(&chosen);
    println!("  dictionary: synonyms of the chosen meaning   {x:>5.1}  ({p:.1}, {r:.1})");
    let (p, r, x) = f(&every);
    println!("  dictionary: synonyms of any meaning          {x:>5.1}  ({p:.1}, {r:.1})");
    for (k, (name, _)) in models.iter().enumerate() {
        let (p, r, x) = f(&own[k]);
        println!("  {name:<44} {x:>5.1}  ({p:.1}, {r:.1})");
    }
    println!("judging: GAP ranking acceptable above unacceptable substitutes");
    for (k, (name, _)) in models.iter().enumerate() {
        println!("  {name:<44} {:.1}  (n={})", 100.0 * gaps[k].iter().sum::<f64>() / gaps[k].len().max(1) as f64, gaps[k].len());
    }
    Ok(())
}
