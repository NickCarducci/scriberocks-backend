//! ScribeIE with its own fine-tuned encoder, on Simple English Wikipedia's
//! sentence edits (scripts/extract_simplewiki_edits.py; CC BY-SA): the
//! encoder starts from a style's full-precision model and learns with the
//! edit tagger's heads (src/edit_tagger.rs). Scored with SARI against
//! copying, on held-out articles' edits.
//!
//!   cargo run --release --features metal --example scribe_ie_finetune \
//!     [base = models/casual-v2] [out = casual] [train sentences = 200000] [phrases = 1000] [epochs = 2]
use scribe::bert::Bert;
use scribe::edit_tagger::{align, apply, sari, EditTagger};
use scribe::text::tokenize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn read(path: &str, cap: usize) -> Vec<(Vec<String>, Vec<String>)> {
    std::fs::read_to_string(path).unwrap().lines().filter_map(|l| {
        let mut c = l.split('\t');
        let (_, b, a) = (c.next()?, c.next()?, c.next()?);
        Some((tokenize(b), tokenize(a)))
    }).filter(|(b, a)| !b.is_empty() && b != a).take(cap).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arg = |i: usize, d: &str| std::env::args().nth(i).unwrap_or_else(|| d.to_string());
    let base = PathBuf::from(arg(1, "models/casual-v2"));
    let out = Path::new("models/scribe-ie").join(arg(2, "casual"));
    let cap: usize = arg(3, "200000").parse()?;
    let k: usize = arg(4, "1000").parse()?;
    let epochs: usize = arg(5, "2").parse()?;
    #[cfg(feature = "metal")]
    let dev = candle_core::Device::new_metal(0)?;
    #[cfg(not(feature = "metal"))]
    let dev = candle_core::Device::Cpu;

    let data = "data/simplewiki-history";
    let train = read(&format!("{data}/train.tsv"), cap);
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (b, a) in &train {
        for ins in align(b, a).1.iter().filter(|i| !i.is_empty() && i.len() <= 4) { *counts.entry(ins.join(" ")).or_default() += 1 }
    }
    let mut phrases: Vec<(String, usize)> = counts.into_iter().collect();
    phrases.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let phrases: Vec<String> = phrases.into_iter().take(k).map(|p| p.0).collect();
    let id: HashMap<&str, usize> = phrases.iter().enumerate().map(|(i, p)| (p.as_str(), i + 1)).collect();
    let examples: Vec<(Vec<String>, Vec<bool>, Vec<usize>)> = train.iter().filter_map(|(b, a)| {
        let (keep, ins) = align(b, a);
        let ids: Option<Vec<usize>> = ins.iter().map(|i| if i.is_empty() { Some(0) } else { id.get(i.join(" ").as_str()).copied() }).collect();
        Some((b.clone(), keep, ids?))
    }).collect();
    println!("{} phrases; {} of {} training edits expressible ({:.0}%)", phrases.len(), examples.len(), train.len(), 100.0 * examples.len() as f64 / train.len().max(1) as f64);

    let (bert, vars) = Bert::load_trainable(&base, dev.clone())?;
    let t = std::time::Instant::now();
    let tagger = EditTagger::train_finetuned(&bert, &vars, &examples, phrases, 512, epochs, 3e-5, 1e-3, 11, &dev)?;
    println!("trained in {:.0?}", t.elapsed());
    let enc = out.join("encoder");
    std::fs::create_dir_all(&enc)?;
    vars.save(enc.join("model.safetensors"))?;
    for f in ["config.json", "vocab.txt"] { std::fs::copy(base.join(f), enc.join(f))?; }
    tagger.save(&out)?;

    let encoder = Bert::load(&enc, dev.clone())?;
    let score = |name: &str, pairs: &[(Vec<String>, Vec<String>)], vs: &[Vec<Vec<f32>>], c: f32| -> f64 {
        let (mut s, mut copy, mut changed, mut exact, mut n) = (0.0, 0.0, 0, 0, 0);
        for ((b, a), v) in pairs.iter().zip(vs) {
            if v.len() != b.len() { continue }
            let (keep, ins) = tagger.tag(v, c).unwrap();
            let pred = apply(b, &keep, &ins);
            s += sari(b, &pred, a);
            copy += sari(b, b, a);
            changed += usize::from(&pred != b);
            exact += usize::from(&pred == a);
            n += 1;
        }
        let nf = n.max(1) as f64;
        println!("  {name:<28} confidence {c:.2}: SARI {:.1} (copying {:.1}); changed {:.0}%, exactly the editor's {:.1}% (n={n})", s / nf, copy / nf, 100.0 * changed as f64 / nf, 100.0 * exact as f64 / nf);
        s / nf
    };
    let vectors = |pairs: &[(Vec<String>, Vec<String>)]| -> Vec<Vec<Vec<f32>>> {
        let sents: Vec<Vec<String>> = pairs.iter().map(|p| p.0.clone()).collect();
        sents.chunks(256).flat_map(|c| encoder.contextual_batch(c).unwrap()).collect()
    };
    let dev_pairs = read(&format!("{data}/dev.tsv"), 5000);
    let dv = vectors(&dev_pairs);
    let mut best = (0.5f32, f64::MIN);
    for c in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] {
        let s = score("dev (held-out articles)", &dev_pairs, &dv, c);
        if s > best.1 { best = (c, s) }
    }
    let test = read(&format!("{data}/test.tsv"), 5000);
    let tv = vectors(&test);
    println!("test, at the dev's best confidence:");
    score("test (held-out articles)", &test, &tv, best.0);
    std::fs::write(out.join("confidence.txt"), format!("{}\n", best.0))?;
    println!("→ {}", out.display());
    Ok(())
}
