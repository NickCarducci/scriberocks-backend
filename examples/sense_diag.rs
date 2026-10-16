//! Diagnose example-based sense picking on the held-out dictionary examples.
use scribe::usage::{corpus, model::{Device, UsageModel}, senses::SenseIndex};
use scribe::{text, Lexicon};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = Path::new("models");
    let lex = Lexicon::load(&models.join("lexicon.bin"))?;
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;
    let name = std::env::args().nth(1).unwrap_or_else(|| "usage".into());
    let model = UsageModel::load(&models.join(&name), device)?;
    println!("== {name}");
    let tok = |s: &str| -> Vec<String> { text::tokenize(s).into_iter().map(|w| w.to_lowercase()).collect() };
    let ex = corpus::wordnet_examples(&lex);
    let locate = |e: &corpus::SenseExample| -> Option<(Vec<String>, usize)> {
        if e.word.contains(' ') { return None }
        let t = tok(&e.sentence);
        let at = corpus::find_word(&t, &e.word, |w| lex.lemmatize(w, None))?;
        (at < 48).then_some((t, at))
    };
    let all: Vec<(bool, u32, Vec<String>, usize)> = ex.iter().filter_map(|e| locate(e).map(|(t, a)| (corpus::is_held_out(e.synset, &e.sentence, 10), e.synset, t, a))).collect();


    // Vectors: word visible, and word masked (context only).
    let mut visible = Vec::new();
    let mut masked = Vec::new();
    for chunk in all.chunks(256) {
        let v: Vec<Vec<String>> = chunk.iter().map(|(_, _, t, _)| t.clone()).collect();
        let m: Vec<Vec<String>> = chunk.iter().map(|(_, _, t, a)| { let mut t = t.clone(); t[*a] = "[MASK]".into(); t }).collect();
        // Keep one vector per example; words past the window (pieces can overflow it) get None.
        for ((_, _, _, a), h) in chunk.iter().zip(model.contextual_batch(&v)?) { visible.push(h.get(*a).cloned()); }
        for ((_, _, _, a), h) in chunk.iter().zip(model.contextual_batch(&m)?) { masked.push(h.get(*a).cloned()); }
    }
    let keep: Vec<bool> = visible.iter().zip(&masked).map(|(v, m)| v.is_some() && m.is_some()).collect();
    let all: Vec<_> = all.into_iter().zip(&keep).filter(|(_, k)| **k).map(|(x, _)| x).collect();
    let _visible: Vec<Vec<f32>> = visible.into_iter().zip(&keep).filter(|(_, k)| **k).filter_map(|(v, _)| v).collect();
    let masked: Vec<Vec<f32>> = masked.into_iter().zip(&keep).filter(|(_, k)| **k).filter_map(|(v, _)| v).collect();
    let mean = |vs: &[Vec<f32>]| -> Vec<f32> {
        let mut m = vec![0.0; vs[0].len()];
        for v in vs { for (a, b) in m.iter_mut().zip(v) { *a += b } }
        m.iter().map(|x| x / vs.len() as f32).collect()
    };
    // Masked + centered vectors; gated override of the first-listed sense.
    let mu = mean(&masked);
    let adj = |v: &Vec<f32>| -> Vec<f32> { v.iter().zip(&mu).map(|(a, b)| a - b).collect() };
    let mut index = SenseIndex::default();
    for ((held, s, _, _), v) in all.iter().zip(masked.iter()) { if !held { index.add(*s, &adj(v)); } }
    // (gold, first, best indexed sense, margin of best over first-listed)
    let mut cases = Vec::new();
    for ((held, s, t, a), v) in all.iter().zip(masked.iter()) {
        if !held { continue }
        let pos = lex.synsets[*s as usize].pos;
        let mut senses = lex.senses(&t[*a], pos);
        if senses.len() < 2 { senses = lex.senses(&ex.iter().find(|e| e.synset == *s).unwrap().word, pos); }
        if senses.len() < 2 { continue }
        let q = adj(v);
        let first = senses[0];
        let first_sim = index.similarity(first, &q);
        let best = senses.iter().filter_map(|&c| index.similarity(c, &q).map(|x| (c, x))).max_by(|x, y| x.1.total_cmp(&y.1));
        let (pick, margin) = match best {
            Some((c, x)) if c != first => (c, x - first_sim.unwrap_or(0.0)),
            _ => (first, 0.0),
        };
        cases.push((*s, first, pick, margin, index.knows(*s)));
    }
    let (tune, report): (Vec<_>, Vec<_>) = cases.iter().enumerate().partition(|(i, _)| i % 2 == 0);
    let acc = |set: &[(usize, &(u32, u32, u32, f32, bool))], tau: f32| -> f64 {
        let hit = set.iter().filter(|(_, (g, f, p, m, _))| (if *m > tau { p } else { f }) == g).count();
        100.0 * hit as f64 / set.len() as f64
    };
    let indexed: Vec<_> = cases.iter().filter(|c| c.4).collect();
    let cond = indexed.iter().filter(|(g, f, p, m, _)| (if *m > 0.0 { p } else { f }) == g).count();
    println!("when the right sense has examples (n={}): override-when-better {:.1}% vs first-listed {:.1}%",
        indexed.len(), 100.0 * cond as f64 / indexed.len() as f64,
        100.0 * indexed.iter().filter(|(g, f, ..)| f == g).count() as f64 / indexed.len() as f64);
    let taus = [0.0f32, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 9.0];
    let best_tau = *taus.iter().max_by(|a, b| acc(&tune, **a).total_cmp(&acc(&tune, **b))).unwrap();
    for t in taus { println!("  tau {t:<4} tune {:.1}%", acc(&tune, t)); }
    println!("chosen tau {best_tau} → report half: gated {:.1}% vs first-listed {:.1}% (n={})", acc(&report, best_tau), acc(&report, 9.0), report.len());
    Ok(())
}
