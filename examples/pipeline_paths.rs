//! One suggestion's path through ScribeSLM, for FidelityAI records: the original
//! word's meaning and confidence, the substitute's meaning and confidence in place,
//! the slot's grammar, and the judge's log-probability of each word in the slot.
//! Input rows as scripts/swords_substitutions.py writes them. One JSON line per row.
//!
//!   cargo run --release --example pipeline_paths -- <in.tsv> <out.jsonl> <judge dir>
use scribe::Engine;
use scribe::bert::Bert;
use serde_json::json;
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let e = Engine::load(Path::new("models"))?;
    let judge = Bert::load(Path::new(&a[3]), scribe::usage::model::Device::Cpu)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[2])?);
    let meaning = |x: Option<u32>| x.map(|x| json!({"s": format!("s{x}"), "c": e.lexicon.category_name(x)}));
    let (mut n, mut skipped) = (0, 0);
    for line in std::fs::read_to_string(&a[1])?.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        let [split, tid, y, prefix, target, sub, suffix] = c[..] else { continue };
        let at = scribe::text::tokenize(prefix).len();
        let before = scribe::text::tokenize(&format!("{prefix} {target} {suffix}"));
        let after = scribe::text::tokenize(&format!("{prefix} {sub} {suffix}"));
        if before.get(at).map(|w| w.eq_ignore_ascii_case(target)) != Some(true) || after.get(at).map(|w| w.eq_ignore_ascii_case(sub)) != Some(true) {
            skipped += 1;
            continue;
        }
        let (pb, pa) = (e.parse_tokens(&before), e.parse_tokens(&after));
        let (o, s) = (&pb.tokens[at], &pa.tokens[at]);
        let low: Vec<String> = before.iter().map(|w| w.to_lowercase()).collect();
        let fit = judge.slot_scores(&low, at, &[target.to_lowercase(), sub.to_lowercase()]).unwrap_or_default();
        let gov = s.head.map(|h| &pa.tokens[h]);
        writeln!(out, "{}", json!({
            "split": split, "tid": tid, "y": y.parse::<u8>()?, "text": after.join(" "),
            "target": o.lemma, "substitute": s.lemma,
            "meaning": meaning(o.sense), "gap": o.sense_gap, "sub_meaning": meaning(s.sense), "sub_gap": s.sense_gap,
            "pos": s.upos, "rel": s.deprel.split(':').next(), "gov_meaning": meaning(gov.and_then(|g| g.sense)),
            "judge": fit,
        }))?;
        n += 1;
    }
    eprintln!("{n} paths, {skipped} skipped");
    Ok(())
}
