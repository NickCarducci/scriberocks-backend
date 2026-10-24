//! Trains ScribeIE's edit tagger (src/edit_tagger.rs) on editors' clarity
//! revisions and scores it with SARI against copying the sentence unchanged.
//!
//! Training: IteraTeR's sentence-level clarity edits from Wikipedia and
//! Wikinews (numeric document ids; arXiv text is left out, as not
//! commercially reusable). Intent labels are IteraTeR's own classifier's.
//! Tested on IteraTeR's test splits (test only, never trained on).
//!
//!   cargo run --release --example scribe_ie_train [style = publishable] [phrases = 500] [epochs = 4] [hidden = 0]
//!   REWORDS=1 trains (and picks the threshold) on rewording edits only: no
//!   name or number changed (`edit_tagger::rewords`); WRITES=1 on edits that insert
//!   or replace a word (`edit_tagger::writes`). OUT names the model folder.
use scribe::bert::Bert;
use scribe::edit_tagger::{align, apply, rewords, sari, writes, EditTagger, Example};
use scribe::text::tokenize;
use std::collections::HashMap;
use std::path::Path;

struct Pair { before: Vec<String>, after: Vec<String> }

fn read(path: &str, clarity_only: bool, non_arxiv: bool) -> Vec<Pair> {
    std::fs::read_to_string(path).unwrap().lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()).filter(|r| {
        (!clarity_only || r["labels"] == "clarity") && (!non_arxiv || r["doc_id"].as_str().is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())))
    }).filter_map(|r| {
        let (b, a) = (tokenize(r["before_sent"].as_str()?.trim()), tokenize(r["after_sent"].as_str()?.trim()));
        (!b.is_empty() && b.len() <= 60 && a.len() <= 70 && b != a).then_some(Pair { before: b, after: a })
    }).collect()
}

fn vectors(bert: &Bert, pairs: &[Pair]) -> Vec<Option<Vec<Vec<f32>>>> {
    let sents: Vec<Vec<String>> = pairs.iter().map(|p| p.before.clone()).collect();
    let mut out = Vec::with_capacity(sents.len());
    for (n, chunk) in sents.chunks(256).enumerate() {
        if n % 20 == 0 { eprintln!("  vectors {}/{}", n * 256, sents.len()) }
        let vs = bert.contextual_batch(chunk).unwrap();
        out.extend(vs.into_iter().zip(chunk).map(|(v, s)| (v.len() == s.len()).then_some(v)));
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let style = std::env::args().nth(1).unwrap_or_else(|| "publishable".into());
    let k: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(500);
    let epochs: usize = std::env::args().nth(3).and_then(|v| v.parse().ok()).unwrap_or(4);
    let hidden: usize = std::env::args().nth(4).and_then(|v| v.parse().ok()).unwrap_or(0);
    let only_rewords = std::env::var_os("REWORDS").is_some();
    // WRITES=1: also only edits that insert or replace a word (no pure deletions).
    let only_writes = std::env::var_os("WRITES").is_some();
    let reworded = |v: Vec<Pair>| -> Vec<Pair> {
        v.into_iter().filter(|p| (!only_rewords || rewords(&p.before, &p.after)) && (!only_writes || writes(&p.before, &p.after))).collect()
    };
    let base = "data/iterater/IteraTeR";
    let bert = Bert::load(&Path::new("models/judges").join(&style), scribe::usage::model::Device::Cpu)?;

    let train = reworded(read(&format!("{base}/full_sent_level/train.json"), true, true));
    let dev = reworded(read(&format!("{base}/full_sent_level/dev.json"), true, true));
    // The phrases editors add most often (up to 4 words).
    let mut counts: HashMap<String, usize> = HashMap::new();
    for p in &train {
        for ins in align(&p.before, &p.after).1.iter().filter(|i| !i.is_empty() && i.len() <= 4) {
            *counts.entry(ins.join(" ")).or_default() += 1;
        }
    }
    let mut phrases: Vec<(String, usize)> = counts.into_iter().collect();
    phrases.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let phrases: Vec<String> = phrases.into_iter().take(k).map(|p| p.0).collect();
    let id: HashMap<&str, usize> = phrases.iter().enumerate().map(|(i, p)| (p.as_str(), i + 1)).collect();

    eprintln!("train: {} clarity edits (Wikipedia + Wikinews)", train.len());
    let vs = vectors(&bert, &train);
    let mut examples = Vec::new();
    for (p, v) in train.iter().zip(vs) {
        let Some(vectors) = v else { continue };
        let (keep, inserts) = align(&p.before, &p.after);
        let ids: Option<Vec<usize>> = inserts.iter().map(|i| if i.is_empty() { Some(0) } else { id.get(i.join(" ").as_str()).copied() }).collect();
        if let Some(inserts) = ids { examples.push(Example { vectors, keep, inserts }) }
    }
    println!("{style}: {} phrases; {} of {} training edits expressible ({:.0}%)", phrases.len(), examples.len(), train.len(), 100.0 * examples.len() as f64 / train.len().max(1) as f64);
    let t = std::time::Instant::now();
    let tagger = EditTagger::train(&examples, phrases, 256, hidden, epochs, 1e-3, 11)?;
    println!("trained in {:.0?}", t.elapsed());
    drop(examples);
    let out = Path::new("models/scribe-ie").join(std::env::var("OUT").unwrap_or_else(|_| style.clone()));
    tagger.save(&out)?;

    // SARI against copying, on edits never trained on.
    let score = |name: &str, pairs: &[Pair], vs: &[Option<Vec<Vec<f32>>>], confidence: f32| {
        let (mut s, mut copy, mut changed, mut exact, mut n) = (0.0, 0.0, 0, 0, 0);
        for (p, v) in pairs.iter().zip(vs) {
            let Some(v) = v else { continue };
            let (keep, ins) = tagger.tag(v, confidence).unwrap();
            let pred = apply(&p.before, &keep, &ins);
            s += sari(&p.before, &pred, &p.after);
            copy += sari(&p.before, &p.before, &p.after);
            changed += usize::from(pred != p.before);
            exact += usize::from(pred == p.after);
            n += 1;
        }
        let n_f = n.max(1) as f64;
        println!("  {name:<44} confidence {confidence:.2}: SARI {:.1} (copying {:.1}); changed {:.0}%, exactly the editor's {:.1}% (n={n})", s / n_f, copy / n_f, 100.0 * changed as f64 / n_f, 100.0 * exact as f64 / n_f);
        s / n_f
    };
    let dv = vectors(&bert, &dev);
    let mut best = (0.5f32, f64::MIN);
    for c in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let s = score("dev (clarity, Wikipedia + Wikinews)", &dev, &dv, c);
        if s > best.1 { best = (c, s) }
    }
    let test = read(&format!("{base}/full_sent_level/test.json"), true, true);
    let test_rewords: Vec<Pair> = read(&format!("{base}/full_sent_level/test.json"), true, true).into_iter().filter(|p| rewords(&p.before, &p.after)).collect();
    let trv = vectors(&bert, &test_rewords);
    let human = read(&format!("{base}/human_sent_level/test.json"), true, false);
    let (tv, hv) = (vectors(&bert, &test), vectors(&bert, &human));
    println!("test, at the dev's best confidence:");
    score("test (clarity, Wikipedia + Wikinews)", &test, &tv, best.0);
    score("test, rewording edits only", &test_rewords, &trv, best.0);
    score("human-labelled test (clarity, all sources)", &human, &hv, best.0);
    std::fs::write(out.join("confidence.txt"), format!("{}\n", best.0))?;
    println!("→ {}", out.display());
    Ok(())
}
