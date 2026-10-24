//! FidelityAI records of word fit: for each SWORDS substitution
//! (scripts/swords_substitutions.py), the meaning the original word has in
//! its sentence, the substitute's own meaning with it in place (both from the
//! sense head), and the slot's grammar: part of speech, relation, and the
//! word it attaches to. Each value can be common alone while the combination
//! is not: a synonym of some other meaning of the word.
//!
//!   cargo run --release --example substitution_records -- <in.tsv> <out.jsonl>
use scribe::engine::Sentence;
use scribe::Engine;
use serde_json::json;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (inp, outp) = (args.get(1).ok_or("usage: substitution_records <in.tsv> <out.jsonl>")?, args.get(2).ok_or("out")?);
    let e = Engine::load(Path::new("models"))?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(outp)?);
    let mut originals: HashMap<String, Option<(Sentence, usize)>> = HashMap::new();
    let (mut n, mut skipped) = (0, 0);
    // The target's token: the first token after the prefix's, when it is the word.
    let locate = |prefix: &str, word: &str, suffix: &str| -> Option<(Vec<String>, usize)> {
        let at = scribe::text::tokenize(prefix).len();
        let tokens = scribe::text::tokenize(&format!("{prefix} {word} {suffix}"));
        (tokens.get(at).map(|t| t.eq_ignore_ascii_case(word)) == Some(true)).then_some((tokens, at))
    };
    for line in std::io::BufReader::new(std::fs::File::open(inp)?).lines() {
        let line = line?;
        let c: Vec<&str> = line.split('\t').collect();
        let [split, tid, y, prefix, target, sub, suffix] = c[..] else { continue };
        let original = originals
            .entry(tid.to_string())
            .or_insert_with(|| locate(prefix, target, suffix).map(|(t, i)| (e.parse_tokens(&t), i)));
        let (Some((orig, oi)), Some((tokens, si))) = (original.as_ref(), locate(prefix, sub, suffix)) else {
            skipped += 1;
            continue;
        };
        let s = e.parse_tokens(&tokens);
        let (ot, st) = (&orig.tokens[*oi], &s.tokens[si]);
        let meaning = |x: Option<u32>| x.map(|x| json!({"s": format!("s{x}"), "c": e.lexicon.category_name(x)}));
        let governor = st.head.map(|h| &s.tokens[h]);
        writeln!(
            out,
            "{}",
            json!({
                "split": split, "tid": tid, "y": y.parse::<u8>()?, "text": tokens.join(" "),
                "target": ot.lemma, "meaning": meaning(ot.sense),
                "substitute": st.lemma, "sub_meaning": meaning(st.sense),
                "pos": st.upos, "rel": st.deprel.split(':').next(),
                "governor": governor.map(|g| g.lemma.clone()), "gov_meaning": meaning(governor.and_then(|g| g.sense)),
            })
        )?;
        n += 1;
    }
    eprintln!("{n} substitutions, {skipped} skipped (target not found as one token)");
    Ok(())
}
