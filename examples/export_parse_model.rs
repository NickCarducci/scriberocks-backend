//! Exports Scriberocks' tagger (UPOS), parser and the dictionary's parts of
//! speech as one compact binary for FidelityAI's JavaScript path B extractor
//! (scribe_parse.js), pruned (weights below 1.0 dropped: no accuracy loss on
//! the English Web Treebank test, see examples/frame_eval.rs).
//!
//!   cargo run --release --example export_parse_model <out.bin> [prune = 1.0]
//!
//! Layout (little-endian): "SCRP" · u32 version · u32 header length · JSON
//! header {tags, tagdict, labels, forms} · tagger table · parser table ·
//! lemma table. A table: u32 n, then n × (u32 hi, u32 lo, u8 nnz, nnz × (u8 class, f32 w)),
//! keyed by key(feature) = (fnv1a32(f) & 0x1FFFFF) · 2^32 + djb2(f), a 53-bit
//! integer both runtimes compute. The lemma table: u32 n, n × (u32 hi, u32 lo,
//! u8 bits) with bits n=1 v=2 a=4 r=8.
//!
//! The models learned from UD English EWT (CC BY-SA 4.0) and Open English
//! WordNet (CC BY 4.0): the exported file carries those terms.
use scribe::lexicon::Pos;
use scribe::Engine;
use std::collections::HashMap;
use std::path::Path;

pub fn fnv1a(s: &str) -> u32 {
    s.bytes().fold(0x811c9dc5u32, |h, b| (h ^ b as u32).wrapping_mul(0x01000193))
}

pub fn djb2(s: &str) -> u32 {
    s.bytes().fold(5381u32, |h, b| h.wrapping_mul(33) ^ b as u32)
}

fn key(s: &str) -> (u32, u32) {
    (fnv1a(s) & 0x1F_FFFF, djb2(s))
}

fn table(out: &mut Vec<u8>, weights: &HashMap<String, Vec<f32>>) -> usize {
    let mut seen: HashMap<(u32, u32), &str> = HashMap::new();
    let mut rows: Vec<(&String, &Vec<f32>)> = weights.iter().filter(|(_, w)| w.iter().any(|&x| x != 0.0)).collect();
    rows.sort_by(|a, b| a.0.cmp(b.0));
    out.extend((rows.len() as u32).to_le_bytes());
    for (f, w) in rows {
        let k = key(f);
        if let Some(other) = seen.insert(k, f) {
            panic!("key collision: {other:?} and {f:?}");
        }
        out.extend(k.0.to_le_bytes());
        out.extend(k.1.to_le_bytes());
        let nz: Vec<(usize, f32)> = w.iter().copied().enumerate().filter(|(_, x)| *x != 0.0).collect();
        out.push(nz.len() as u8);
        for (c, x) in nz {
            out.push(c as u8);
            out.extend(x.to_le_bytes());
        }
    }
    seen.len()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_path = std::env::args().nth(1).ok_or("usage: export_parse_model <out.bin> [prune]")?;
    let prune: f32 = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let mut e = Engine::load(Path::new("models"))?;
    e.upos.prune(prune);
    e.parser.prune(prune);
    let (tags, tagdict, tw) = e.upos.parts();
    let (labels, pw) = e.parser.parts();
    assert!(tags.len() <= 256 && 2 + 2 * labels.len() <= 256, "classes fit a byte");
    let header = serde_json::json!({
        "tags": tags,
        "tagdict": tagdict,
        "labels": labels,
        "forms": e.lexicon.forms,
        "prune": prune,
        "license": "Weights learned from UD English EWT (CC BY-SA 4.0) and Open English WordNet (CC BY 4.0); this file is shared under CC BY-SA 4.0.",
    })
    .to_string();
    let mut out = b"SCRP".to_vec();
    out.extend(1u32.to_le_bytes());
    out.extend((header.len() as u32).to_le_bytes());
    out.extend(header.as_bytes());
    let nt = table(&mut out, tw);
    let np = table(&mut out, pw);
    let mut lemmas: Vec<(&String, u8)> = e.lexicon.lemmas.iter().map(|(l, ps)| {
        (l, ps.iter().fold(0u8, |b, (p, _)| b | match p { Pos::Noun => 1, Pos::Verb => 2, Pos::Adj => 4, Pos::Adv => 8 }))
    }).collect();
    lemmas.sort();
    out.extend((lemmas.len() as u32).to_le_bytes());
    let mut seen: HashMap<(u32, u32), &str> = HashMap::new();
    for (l, bits) in &lemmas {
        let k = key(l);
        if let Some(other) = seen.insert(k, l) { panic!("lemma key collision: {other:?} and {l:?}") }
        out.extend(k.0.to_le_bytes());
        out.extend(k.1.to_le_bytes());
        out.push(*bits);
    }
    std::fs::write(&out_path, &out)?;
    println!("{out_path}: {:.1} MB (header {:.1} MB; tagger {nt} features, parser {np} features, {} lemmas, {} irregular forms)",
        out.len() as f64 / 1e6, header.len() as f64 / 1e6, lemmas.len(), e.lexicon.forms.len());
    Ok(())
}
