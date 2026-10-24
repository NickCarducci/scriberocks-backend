//! Dependency parser: arc-eager transitions chosen by an averaged perceptron,
//! trained from scratch on UD English EWT (projective sentences, static oracle).
//!
//! Output is a tree: every word gets a head word and a relation
//! (nsubj, obj, aux, advcl, mark, …).

use crate::conllu::Sentence;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Move {
    Shift,
    Reduce,
    Left(u16),
    Right(u16),
}

#[derive(Default)]
pub struct Parser {
    labels: Vec<String>,
    weights: HashMap<String, Vec<f32>>,
}

/// On-disk form: only non-negligible weights, as (class, weight) pairs.
#[derive(Serialize, Deserialize)]
struct Stored {
    labels: Vec<String>,
    weights: Vec<(String, Vec<(u16, f32)>)>,
}

#[derive(Clone, Debug)]
pub struct Arc {
    /// 0 = root, otherwise 1-based word index.
    pub head: usize,
    pub label: String,
}

struct State<'a> {
    words: &'a [String],
    tags: &'a [String],
    stack: Vec<usize>,
    next: usize,
    heads: Vec<Option<usize>>,
    labels: Vec<u16>,
    left: Vec<Vec<usize>>,
    right: Vec<Vec<usize>>,
}

impl<'a> State<'a> {
    /// `words`/`tags` include a ROOT placeholder at index 0.
    fn new(words: &'a [String], tags: &'a [String]) -> Self {
        let n = words.len();
        State { words, tags, stack: vec![0], next: 1, heads: vec![None; n], labels: vec![0; n], left: vec![vec![]; n], right: vec![vec![]; n] }
    }
    fn done(&self) -> bool {
        self.next >= self.words.len()
    }
    fn valid(&self, m: Move) -> bool {
        let s0 = self.stack.last().copied();
        match m {
            Move::Shift => !self.done(),
            Move::Reduce => s0.is_some_and(|s| s != 0 && self.heads[s].is_some()),
            Move::Left(_) => !self.done() && s0.is_some_and(|s| s != 0 && self.heads[s].is_none()),
            Move::Right(_) => !self.done() && s0.is_some(),
        }
    }
    fn apply(&mut self, m: Move) {
        match m {
            Move::Shift => {
                self.stack.push(self.next);
                self.next += 1;
            }
            Move::Reduce => {
                self.stack.pop();
            }
            Move::Left(l) => {
                let s = self.stack.pop().unwrap();
                self.heads[s] = Some(self.next);
                self.labels[s] = l;
                self.left[self.next].push(s);
            }
            Move::Right(l) => {
                let s = *self.stack.last().unwrap();
                let b = self.next;
                self.heads[b] = Some(s);
                self.labels[b] = l;
                self.right[s].push(b);
                self.stack.push(b);
                self.next += 1;
            }
        }
    }

    fn features(&self, labels: &[String]) -> Vec<String> {
        let w = |i: Option<usize>| i.map_or("-", |i| self.words[i].as_str());
        let t = |i: Option<usize>| i.map_or("-", |i| self.tags[i].as_str());
        let lab = |i: Option<usize>| i.map_or("-", |i| labels[self.labels[i] as usize].as_str());
        let s0 = self.stack.last().copied();
        let s1 = self.stack.len().checked_sub(2).map(|i| self.stack[i]);
        let n = self.words.len();
        let n0 = (self.next < n).then_some(self.next);
        let n1 = (self.next + 1 < n).then_some(self.next + 1);
        let n2 = (self.next + 2 < n).then_some(self.next + 2);
        let s0h = s0.and_then(|s| self.heads[s]);
        let s0l = s0.and_then(|s| self.left[s].first().copied());
        let s0r = s0.and_then(|s| self.right[s].last().copied());
        let n0l = n0.and_then(|b| self.left[b].first().copied());
        let dist = match (s0, n0) {
            (Some(s), Some(b)) => (b - s).min(5).to_string(),
            _ => "-".into(),
        };
        let vs0 = s0.map_or(0, |s| self.right[s].len().min(3));
        let vn0 = n0.map_or(0, |b| self.left[b].len().min(3));
        vec![
            "bias".into(),
            format!("s0w {}", w(s0)),
            format!("s0t {}", t(s0)),
            format!("s0wt {} {}", w(s0), t(s0)),
            format!("n0w {}", w(n0)),
            format!("n0t {}", t(n0)),
            format!("n0wt {} {}", w(n0), t(n0)),
            format!("n1w {}", w(n1)),
            format!("n1t {}", t(n1)),
            format!("n2t {}", t(n2)),
            format!("s1t {}", t(s1)),
            format!("s0t n0t {} {}", t(s0), t(n0)),
            format!("s0w n0w {} {}", w(s0), w(n0)),
            format!("s0wt n0t {} {} {}", w(s0), t(s0), t(n0)),
            format!("s0t n0wt {} {} {}", t(s0), w(n0), t(n0)),
            format!("s0t n0t n1t {} {} {}", t(s0), t(n0), t(n1)),
            format!("s1t s0t n0t {} {} {}", t(s1), t(s0), t(n0)),
            format!("s0ht s0t n0t {} {} {}", t(s0h), t(s0), t(n0)),
            format!("s0lt s0t n0t {} {} {}", lab(s0l), t(s0), t(n0)),
            format!("s0rt s0t n0t {} {} {}", lab(s0r), t(s0), t(n0)),
            format!("n0lt n0t {} {}", lab(n0l), t(n0)),
            format!("dist s0t n0t {} {} {}", dist, t(s0), t(n0)),
            format!("dist s0w n0w {} {} {}", dist, w(s0), w(n0)),
            format!("val s0 {} {}", t(s0), vs0),
            format!("val n0 {} {}", t(n0), vn0),
            format!("s0has {}", s0.is_some_and(|s| self.heads[s].is_some())),
            format!("s1w {}", w(s1)),
            format!("s0hw {}", w(s0h)),
            format!("n0suf {}", suffix3(w(n0))),
            format!("s0suf {}", suffix3(w(s0))),
            format!("s0l s0r {} {} {}", lab(s0l), lab(s0r), t(s0)),
            format!("n0t n1t n2t {} {} {}", t(n0), t(n1), t(n2)),
        ]
    }
}

impl Parser {
    /// Drops weights smaller than `threshold` in magnitude, and features left
    /// with none (for shipping a smaller model). Returns (features, weights) kept.
    pub fn prune(&mut self, threshold: f32) -> (usize, usize) {
        self.weights.retain(|_, w| {
            for x in w.iter_mut() { if x.abs() < threshold { *x = 0.0 } }
            w.iter().any(|&x| x != 0.0)
        });
        (self.weights.len(), self.weights.values().map(|w| w.iter().filter(|&&x| x != 0.0).count()).sum())
    }

    /// Labels and every feature's weights over the 2 + 2·labels moves
    /// (shift, reduce, left-arc per label, right-arc per label), for
    /// exporting the parser to other runtimes.
    pub fn parts(&self) -> (&[String], &HashMap<String, Vec<f32>>) {
        (&self.labels, &self.weights)
    }

    pub fn load(path: &Path) -> Result<Parser, Box<dyn std::error::Error>> {
        let stored: Stored = postcard::from_bytes(&std::fs::read(path)?)?;
        let classes = 2 + 2 * stored.labels.len();
        let weights = stored
            .weights
            .into_iter()
            .map(|(f, sparse)| {
                let mut w = vec![0.0; classes];
                for (c, v) in sparse {
                    w[c as usize] = v;
                }
                (f, w)
            })
            .collect();
        Ok(Parser { labels: stored.labels, weights })
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let weights = self
            .weights
            .iter()
            .map(|(f, w)| (f.clone(), w.iter().enumerate().filter(|(_, v)| v.abs() > 0.02).map(|(c, v)| (c as u16, *v)).collect::<Vec<_>>()))
            .filter(|(_, sparse)| !sparse.is_empty())
            .collect();
        std::fs::write(path, postcard::to_allocvec(&Stored { labels: self.labels.clone(), weights })?)?;
        Ok(())
    }

    fn classes(&self) -> usize {
        2 + 2 * self.labels.len()
    }

    fn class_of(&self, m: Move) -> usize {
        match m {
            Move::Shift => 0,
            Move::Reduce => 1,
            Move::Left(l) => 2 + l as usize,
            Move::Right(l) => 2 + self.labels.len() + l as usize,
        }
    }

    fn move_of(&self, c: usize) -> Move {
        let l = self.labels.len();
        match c {
            0 => Move::Shift,
            1 => Move::Reduce,
            c if c < 2 + l => Move::Left((c - 2) as u16),
            c => Move::Right((c - 2 - l) as u16),
        }
    }

    fn scores(&self, feats: &[String]) -> Vec<f32> {
        let mut scores = vec![0f32; self.classes()];
        for f in feats {
            if let Some(w) = self.weights.get(f) {
                for (s, v) in scores.iter_mut().zip(w) {
                    *s += v;
                }
            }
        }
        scores
    }

    fn best_valid(&self, state: &State, scores: &[f32]) -> Move {
        let mut best: Option<(usize, f32)> = None;
        for (c, &s) in scores.iter().enumerate() {
            if state.valid(self.move_of(c)) && best.is_none_or(|(_, b)| s > b) {
                best = Some((c, s));
            }
        }
        self.move_of(best.map_or(0, |(c, _)| c))
    }

    /// Parses tokens with their UPOS tags. Returns one arc per word.
    pub fn parse(&self, words: &[&str], tags: &[&str]) -> Vec<Arc> {
        let w: Vec<String> = std::iter::once("<root>".to_string()).chain(words.iter().map(|w| w.to_lowercase())).collect();
        let t: Vec<String> = std::iter::once("ROOT".to_string()).chain(tags.iter().map(|t| t.to_string())).collect();
        let mut state = State::new(&w, &t);
        while !state.done() {
            let scores = self.scores(&state.features(&self.labels));
            let m = self.best_valid(&state, &scores);
            state.apply(m);
        }
        // Words left without a head attach to the first root-level word.
        let root_word = (1..w.len()).find(|&i| state.heads[i] == Some(0));
        (1..w.len())
            .map(|i| match state.heads[i] {
                Some(h) => Arc { head: h, label: self.labels[state.labels[i] as usize].clone() },
                None => match root_word {
                    Some(r) if r != i => Arc { head: r, label: "dep".into() },
                    _ => Arc { head: 0, label: "root".into() },
                },
            })
            .collect()
    }

    /// `train_tags[i]` are the (predicted) UPOS tags used as input for
    /// sentence i, so training sees the same tag noise as real use.
    pub fn train(sentences: &[Sentence], train_tags: &[Vec<String>], iterations: usize) -> Parser {
        let mut labels: Vec<String> = sentences.iter().flatten().map(|t| base(&t.deprel).to_string()).collect();
        labels.sort();
        labels.dedup();
        let mut parser = Parser { labels, weights: HashMap::new() };
        let label_index: HashMap<String, u16> = parser.labels.iter().enumerate().map(|(i, l)| (l.clone(), i as u16)).collect();

        let classes = parser.classes();
        let mut totals: HashMap<String, Vec<f64>> = HashMap::new();
        let mut stamps: HashMap<String, Vec<u32>> = HashMap::new();
        let mut step: u32 = 0;
        let mut order: Vec<usize> = (0..sentences.len()).collect();
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut next_rand = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };

        for iter in 0..iterations {
            let (mut correct, mut total) = (0usize, 0usize);
            for &si in &order {
                let s = &sentences[si];
                let words: Vec<String> = std::iter::once("<root>".to_string()).chain(s.iter().map(|t| t.form.to_lowercase())).collect();
                let tags: Vec<String> = std::iter::once("ROOT".to_string()).chain(train_tags[si].iter().cloned()).collect();
                let gold_head: Vec<usize> = std::iter::once(0).chain(s.iter().map(|t| t.head)).collect();
                let gold_label: Vec<u16> = std::iter::once(0).chain(s.iter().map(|t| label_index[base(&t.deprel)])).collect();
                let mut state = State::new(&words, &tags);
                while !state.done() {
                    let feats = state.features(&parser.labels);
                    let scores = parser.scores(&feats);
                    let guess = parser.best_valid(&state, &scores);

                    // Dynamic oracle: every valid move with minimal cost is "correct".
                    let mut min_cost = usize::MAX;
                    let mut costs = vec![usize::MAX; classes];
                    for c in 0..classes {
                        let m = parser.move_of(c);
                        if state.valid(m) {
                            costs[c] = cost(&state, m, &gold_head, &gold_label);
                            min_cost = min_cost.min(costs[c]);
                        }
                    }
                    let best_gold = (0..classes)
                        .filter(|&c| costs[c] == min_cost)
                        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
                        .unwrap();
                    let g = parser.move_of(best_gold);
                    step += 1;
                    total += 1;
                    if costs[parser.class_of(guess)] != min_cost {
                        let (gc, pc) = (best_gold, parser.class_of(guess));
                        for f in &feats {
                            let w = parser.weights.entry(f.clone()).or_insert_with(|| vec![0.0; classes]);
                            let tot = totals.entry(f.clone()).or_insert_with(|| vec![0.0; classes]);
                            let st = stamps.entry(f.clone()).or_insert_with(|| vec![0; classes]);
                            for (c, d) in [(gc, 1.0f32), (pc, -1.0)] {
                                tot[c] += (step - st[c]) as f64 * w[c] as f64;
                                st[c] = step;
                                w[c] += d;
                            }
                        }
                    } else {
                        correct += 1;
                    }
                    // Explore: after the first pass, usually follow our own (possibly wrong) move.
                    let explore = iter > 0 && next_rand() % 10 != 0;
                    state.apply(if explore { guess } else { g });
                }
            }
            eprintln!("  iteration {}: move accuracy {:.2}%", iter + 1, 100.0 * correct as f64 / total.max(1) as f64);
            for i in (1..order.len()).rev() {
                order.swap(i, (next_rand() % (i as u64 + 1)) as usize);
            }
        }
        for (feat, w) in parser.weights.iter_mut() {
            let (Some(tot), Some(st)) = (totals.get(feat), stamps.get(feat)) else { continue };
            for c in 0..classes {
                w[c] = ((tot[c] + (step - st[c]) as f64 * w[c] as f64) / step.max(1) as f64) as f32;
            }
        }
        parser
    }

    /// Unlabeled / labeled attachment scores, using the given tags.
    pub fn evaluate(&self, sentences: &[Sentence], tag: &dyn Fn(&[&str]) -> Vec<String>) -> (f64, f64) {
        let (mut uas, mut las, mut n) = (0usize, 0usize, 0usize);
        for s in sentences {
            let words: Vec<&str> = s.iter().map(|t| t.form.as_str()).collect();
            let tags = tag(&words);
            let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
            for (arc, gold) in self.parse(&words, &tag_refs).iter().zip(s) {
                if arc.head == gold.head {
                    uas += 1;
                    las += (arc.label == base(&gold.deprel)) as usize;
                }
                n += 1;
            }
        }
        (100.0 * uas as f64 / n as f64, 100.0 * las as f64 / n as f64)
    }
}

fn suffix3(w: &str) -> &str {
    let start = w.char_indices().rev().nth(2).map(|(i, _)| i).unwrap_or(0);
    &w[start..]
}

/// "nsubj:pass" → "nsubj"
fn base(deprel: &str) -> &str {
    deprel.split(':').next().unwrap_or(deprel)
}

/// Arc-eager dynamic oracle (Goldberg & Nivre 2012): how many gold arcs
/// become unreachable if `m` is taken now (+1 for a wrong label on a right arc).
fn cost(state: &State, m: Move, head: &[usize], label: &[u16]) -> usize {
    let stack = &state.stack;
    let b = state.next;
    let buffer = b..state.words.len();
    let unattached = |k: usize| state.heads[k].is_none();
    match m {
        Move::Shift => stack.iter().filter(|&&k| head[b] == k || (k != 0 && head[k] == b && unattached(k))).count(),
        Move::Reduce => {
            let s = *stack.last().unwrap();
            buffer.filter(|&k| head[k] == s).count()
        }
        Move::Left(l) => {
            let s = *stack.last().unwrap();
            let lost_head = buffer.clone().filter(|&k| k != b && head[s] == k).count();
            let lost_deps = buffer.filter(|&k| head[k] == s).count();
            lost_head + lost_deps + (head[s] == b && label[s] != l) as usize
        }
        Move::Right(l) => {
            let s = *stack.last().unwrap();
            let lost_head = stack.iter().filter(|&&k| k != s && head[b] == k).count() + buffer.clone().filter(|&k| head[b] == k).count();
            let lost_deps = stack.iter().filter(|&&k| k != 0 && head[k] == b && unattached(k)).count();
            lost_head + lost_deps + (head[b] == s && label[b] != l) as usize
        }
    }
}
