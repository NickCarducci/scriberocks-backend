//! Does ranking swaps by how far their meaning moved put better suggestions first?
//! The real pipeline on human-judged sentences: for each target, its own suggestions
//! are matched to the substitutes people accepted or rejected there.
//!   earlier  an accepted suggestion comes before a rejected one (0.5 chance)
//!   top n    accepted / (accepted + rejected) among those shown in the first n
//!
//!   cargo run --release --example swap_rank_eval -- <subs.tsv> <casual|publishable> <weights> [targets = 300]
use scribe::engine::ProseStyle;
use scribe::lexicon::Pos;
use scribe::{Engine, text};
use std::collections::BTreeMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let weights: Vec<f64> = a[3].split(',').map(|w| w.parse()).collect::<Result<_, _>>()?;
    let n: usize = a.get(4).map(|s| s.parse()).transpose()?.unwrap_or(300);
    let mut e = Engine::load(Path::new("models"))?;
    let options = scribe::scribe_ie::EditOptions { style: ProseStyle::from_name(&a[2]), ..Default::default() };
    let lemmas = |e: &Engine, w: &str| -> Vec<String> {
        let mut out = vec![w.to_lowercase()];
        for p in [Pos::Noun, Pos::Verb, Pos::Adj, Pos::Adv] {
            out.extend(e.lexicon.lemmatize(w, Some(p)));
        }
        out
    };
    // Target: its sentence, and each judged substitute's dictionary forms.
    let mut targets: BTreeMap<String, (String, usize, Vec<(Vec<String>, bool)>)> = BTreeMap::new();
    for line in std::fs::read_to_string(&a[1])?.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let [_, tid, y, prefix, target, sub, suffix] = c[..] else { continue };
        let t = targets.entry(tid.to_string()).or_insert_with(|| (format!("{prefix} {target} {suffix}"), text::tokenize(prefix).len(), vec![]));
        t.2.push((lemmas(&e, sub), y == "1"));
    }
    let items: Vec<_> = targets.into_values().take(n).collect();
    for w in weights {
        e.swap_meaning_weight = w;
        let (mut wins, mut pairs, mut matched) = (0.0, 0, 0);
        let mut top = [(0, 0), (0, 0)];
        for (rock, at, judged) in &items {
            let tokens = text::tokenize(rock);
            let mut seen: Vec<(usize, bool)> = vec![];
            for (rank, v) in e.variants_with(rock, 100, &options).iter().enumerate() {
                let t = text::tokenize(&v.text);
                if t.len() != tokens.len() || t.get(*at).is_none_or(|x| x.eq_ignore_ascii_case(&tokens[*at])) {
                    continue;
                }
                if t.iter().zip(&tokens).filter(|(x, y)| !x.eq_ignore_ascii_case(y)).count() != 1 {
                    continue;
                }
                let forms = lemmas(&e, &t[*at]);
                if let Some((_, ok)) = judged.iter().find(|(l, _)| l.iter().any(|x| forms.contains(x))) {
                    seen.push((rank, *ok));
                }
            }
            matched += seen.len();
            for &(r, ok) in &seen {
                for (slot, limit) in top.iter_mut().zip([5, 12]) {
                    if r < limit {
                        if ok { slot.0 += 1 } else { slot.1 += 1 }
                    }
                }
            }
            for &(ra, oka) in &seen {
                for &(rb, okb) in &seen {
                    if oka && !okb {
                        pairs += 1;
                        if ra < rb { wins += 1.0 }
                    }
                }
            }
        }
        let share = |(x, y): (i32, i32)| x as f64 / (x + y).max(1) as f64;
        println!("weight {w}: {} targets, {matched} matched suggestions; earlier {:.3} of {pairs}; top 5 {:.3} ({}+{}); top 12 {:.3} ({}+{})",
            items.len(), wins / pairs.max(1) as f64, share(top[0]), top[0].0, top[0].1, share(top[1]), top[1].0, top[1].1);
    }
    Ok(())
}
