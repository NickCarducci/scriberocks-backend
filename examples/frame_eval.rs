//! Scriberocks' tagger + parser as a frame extractor, scored like FidelityAI's
//! path B extractor (FidelityAI engine/scripts/frame_extraction_eval.mjs): on
//! UD English Web Treebank test sentences whose root is a verb, the root's
//! lemma (action) and its nsubj / obj lemmas (subject, object). Gold tokens.
//!
//!   cargo run --release --example frame_eval [conllu = data/en_ewt-ud-test.conllu] [prune thresholds, e.g. 0,1,2,4]
//! With thresholds: the tagger's and parser's weights below each are dropped
//! first, reporting what's kept (for shipping a smaller extractor).
use scribe::{conllu, Engine};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "data/en_ewt-ud-test.conllu".into());
    let thresholds: Vec<f32> = std::env::args().nth(2).map_or(vec![0.0], |t| t.split(',').filter_map(|x| x.parse().ok()).collect());
    let gold_all = conllu::parse(&std::fs::read_to_string(&path)?);
    for th in thresholds {
    let mut e = Engine::load(Path::new("models"))?;
    let (tf, tw) = e.upos.prune(th);
    let (pf, pw) = e.parser.prune(th);
    println!("threshold {th}: tagger {tf} features / {tw} weights; parser {pf} features / {pw} weights");
    let (mut n, mut tally) = (0, [(0, 0); 3]);
    for gold in gold_all.clone() {
        let Some(root) = gold.iter().position(|t| t.head == 0) else { continue };
        if gold[root].upos != "VERB" { continue }
        n += 1;
        let kid = |toks: &[conllu::Token], r: &str| toks.iter().position(|t| t.head == root + 1 && t.deprel.split(':').next() == Some(r));
        let want = [Some(root), kid(&gold, "nsubj"), kid(&gold, "obj")].map(|i| i.map(|i| gold[i].lemma.to_lowercase()));
        let words: Vec<String> = gold.iter().map(|t| t.form.clone()).collect();
        let s = e.parse_tokens(&words);
        let pr = s.root();
        let pkid = |r: &str| pr.and_then(|p| s.children(p).find(|&k| s.tokens[k].deprel.split(':').next() == Some(r)));
        let got = [pr, pkid("nsubj"), pkid("obj")].map(|i| i.map(|i| s.tokens[i].lemma.to_lowercase()));
        for slot in 0..3 {
            if let Some(w) = &want[slot] {
                tally[slot].1 += 1;
                if got[slot].as_ref() == Some(w) { tally[slot].0 += 1 }
            }
        }
    }
    println!("{n} sentences with a verb root (gold tokens)");
    for (name, (ok, total)) in ["subject", "action", "object"].iter().zip([tally[1], tally[0], tally[2]]) {
        println!("  {name:<8} {:.1}% correct (n={total})", 100.0 * ok as f64 / total.max(1) as f64);
    }
    }
    Ok(())
}
