//! Synonym sets with shared usage samples (see src/usage/synonyms.rs), built
//! from WordNet synsets and the usage corpus, scored by the usage model.
//!
//!   cargo run --release --example synonym_sets [synsets] [per-member] [margin] [skip]
//!
//! Writes data/synonyms/shared.jsonl: one set per line with its shared
//! sentences, and prints coverage plus a few examples.
use scribe::usage::synonyms;
use scribe::{text, Engine, Pos};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, d: f64| args.get(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (n_synsets, per_member, margin, skip) = (arg(1, 300.0) as usize, arg(2, 10.0) as usize, arg(3, 2.0) as f32, arg(4, 0.0) as usize);
    let e = Engine::load(Path::new("models")).unwrap();
    let usage = e.usage.as_ref().unwrap();

    // Usage corpus: Wiktionary + WordNet examples + EWT, lowercased tokens.
    let tokenize = |s: &str| -> Vec<String> { text::tokenize(s).into_iter().map(|w| w.to_lowercase()).collect() };
    let mut corpus: Vec<Vec<String>> = std::fs::read_to_string("data/wiktionary/examples.txt").unwrap().lines().map(tokenize).collect();
    corpus.extend(e.lexicon.synsets.iter().flat_map(|s| s.examples.iter().map(|x| tokenize(x))));
    let ewt = scribe::conllu::parse(&std::fs::read_to_string("data/en_ewt-ud-train.conllu").unwrap());
    corpus.extend(ewt.iter().map(|s| s.iter().map(|t| t.form.to_lowercase()).collect()));

    // SemCor's even (training) documents, with gold senses: shared samples
    // drawn from them measure precision (is the gold sense the set's meaning?).
    let wsd = "data/semcor/wsdeval/WSD_Evaluation_Framework/Training_Corpora/SemCor";
    let semcor = scribe::semcor::parse(&std::fs::read_to_string(format!("{wsd}/semcor.data.xml")).unwrap(), &std::fs::read_to_string(format!("{wsd}/semcor.gold.key.txt")).unwrap());
    let keys = scribe::semcor::sense_keys(Path::new("data/oewn"), &e.lexicon).unwrap();
    let docs: Vec<&str> = semcor.iter().map(|s| s.document.as_str()).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    let held: std::collections::BTreeSet<&str> = docs.iter().skip(1).step_by(2).copied().collect();
    let mut gold: HashMap<(usize, usize), u32> = HashMap::new();
    for sent in semcor.iter().filter(|s| !held.contains(s.document.as_str())) {
        let at = corpus.len();
        for inst in &sent.instances {
            if let Some(&g) = inst.keys.iter().find_map(|k| keys.get(k)) { gold.insert((at, inst.index), g); }
        }
        corpus.push(sent.words.iter().map(|w| w.to_lowercase()).collect());
    }
    // SemCor first in the occurrence search, so the precision sample is large.
    let semcor_start = corpus.len() - semcor.iter().filter(|s| !held.contains(s.document.as_str())).count();

    // Candidate sets: WordNet synsets with 2+ usable single-word members. The
    // 100 commonest words (mostly function words) are never offered.
    let stop = |w: &str| (usage.vocab.id(w) as usize) < scribe::usage::vocab::SPECIALS.len() + 100;
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for s in &corpus { for w in s { *freq.entry(w.as_str()).or_default() += 1 } }
    let mut candidates: Vec<(u32, Vec<String>, usize)> = e
        .lexicon
        .synsets
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            let members: Vec<String> = s.members.iter().map(|m| m.to_lowercase()).filter(|m| !m.contains(' ') && synonyms::usable_member(m, &stop)).collect();
            (members.len() >= 2).then(|| {
                let least = members.iter().map(|m| freq.get(m.as_str()).copied().unwrap_or(0)).min().unwrap_or(0);
                (i as u32, members, least)
            })
        })
        .collect();
    candidates.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    let picked: Vec<(u32, Vec<String>)> = candidates.iter().skip(skip).take(n_synsets).map(|(s, m, _)| (*s, m.clone())).collect();
    eprintln!("{} candidate sets; building {}", candidates.len(), picked.len());

    // Occurrences of each (member, part of speech), any inflection, tagged so
    // other members can be put in the same form. Up to per_member each.
    let t = Instant::now();
    let pos_of = |xpos: &str| match xpos {
        "NN" | "NNS" => Some(Pos::Noun),
        x if x.starts_with("VB") => Some(Pos::Verb),
        x if x.starts_with("JJ") => Some(Pos::Adj),
        x if x.starts_with("RB") => Some(Pos::Adv),
        _ => None,
    };
    let mut wanted: HashMap<(String, Pos), Vec<synonyms::Occurrence>> = HashMap::new();
    for (s, members) in &picked {
        for m in members { wanted.entry((m.clone(), e.lexicon.synsets[*s as usize].pos)).or_default(); }
    }
    let lemmas_wanted: std::collections::HashSet<String> = wanted.keys().map(|(m, _)| m.clone()).collect();
    let mut open = wanted.len();
    let order: Vec<usize> = (semcor_start..corpus.len()).chain(0..semcor_start).collect();
    for &si in &order {
        let sentence = &corpus[si];
        if open == 0 { break }
        let hit = sentence.iter().any(|w| e.lexicon.lemmatize(w, None).iter().any(|l| lemmas_wanted.contains(l)));
        if !hit { continue }
        let words: Vec<&str> = sentence.iter().map(String::as_str).collect();
        let tags = e.xpos.tag(&words, Some(&e.lexicon));
        for (slot, (w, tag)) in words.iter().zip(&tags).enumerate() {
            let Some(p) = pos_of(tag) else { continue };
            for lemma in e.lexicon.lemmatize(w, Some(p)) {
                if let Some(list) = wanted.get_mut(&(lemma, p)) {
                    if list.len() < per_member && list.last().map(|o| o.sentence) != Some(si) {
                        list.push(synonyms::Occurrence { sentence: si, slot, xpos: tag });
                        if list.len() == per_member { open -= 1 }
                    }
                }
            }
        }
    }
    eprintln!("occurrences found in {:.1?}", t.elapsed());

    let t = Instant::now();
    let occurrences = |synset: u32, member: &str| wanted.get(&(member.to_string(), e.lexicon.synsets[synset as usize].pos)).cloned().unwrap_or_default();
    let inflect = |lemma: &str, xpos: &str| e.inflector.inflect(lemma, xpos);
    let sets = synonyms::build(&picked, &corpus, &occurrences, &inflect, usage as &dyn scribe::engine::SlotScorer, per_member, margin);
    let with = sets.iter().filter(|s| !s.samples.is_empty()).count();
    let samples: usize = sets.iter().map(|s| s.samples.len()).sum();
    eprintln!("{:.1?}: {with}/{} sets have shared samples; {samples} shared samples", t.elapsed(), sets.len());
    let gold_ref = &gold;

    // Reflective judge: a member's profile is the mean context vector (word
    // hidden) over its own uses; a use of member a is a shared sample for b
    // when its context is close to b's profile (centered cosine >= theta).
    // Same occurrences and the same SemCor gold, so precision is comparable.
    let t = Instant::now();
    let mut vectors: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
    let occs: Vec<synonyms::Occurrence> = wanted.values().flatten().copied().collect();
    for chunk in occs.chunks(256) {
        let masked: Vec<Vec<String>> = chunk.iter().map(|o| { let mut m = corpus[o.sentence].clone(); m[o.slot] = "[MASK]".into(); m }).collect();
        for (o, h) in chunk.iter().zip(usage.contextual_batch(&masked).unwrap()) {
            if let Some(v) = h.get(o.slot) { vectors.insert((o.sentence, o.slot), v.clone()); }
        }
    }
    let dim = vectors.values().next().map_or(0, Vec::len);
    let mut center = vec![0f32; dim];
    for v in vectors.values() { for (c, x) in center.iter_mut().zip(v) { *c += x / vectors.len() as f32 } }
    let centered = |v: &[f32]| -> Vec<f32> { v.iter().zip(&center).map(|(a, b)| a - b).collect() };
    let mut profile: HashMap<(String, Pos), Vec<f32>> = HashMap::new();
    for (key, list) in &wanted {
        let vs: Vec<Vec<f32>> = list.iter().filter_map(|o| vectors.get(&(o.sentence, o.slot))).map(|v| centered(v)).collect();
        if vs.is_empty() { continue }
        let mut mean = vec![0f32; dim];
        for v in &vs { for (m, x) in mean.iter_mut().zip(v) { *m += x / vs.len() as f32 } }
        profile.insert(key.clone(), mean);
    }
    eprintln!("reflective profiles for {} members in {:.1?}", profile.len(), t.elapsed());
    for theta in [0.1f32, 0.2, 0.3, 0.4] {
        let (mut kept, mut right) = (0usize, 0usize);
        for (syn, members) in &picked {
            let pos = e.lexicon.synsets[*syn as usize].pos;
            for a in members {
                for o in occurrences(*syn, a).into_iter().take(per_member) {
                    let Some(g) = gold.get(&(o.sentence, o.slot)) else { continue };
                    let Some(v) = vectors.get(&(o.sentence, o.slot)) else { continue };
                    let v = centered(v);
                    // Some other member's own usage looks like this context.
                    let shared = members.iter().filter(|b| *b != a).any(|b| {
                        profile.get(&(b.clone(), pos)).is_some_and(|p| scribe::usage::senses::cosine(p, &v) >= theta)
                    });
                    if shared { kept += 1; right += (*g == *syn) as usize; }
                }
            }
        }
        println!("reflective theta {theta}: precision {:.1}% of {kept} kept SemCor uses", 100.0 * right as f64 / kept.max(1) as f64);
    }
    let judged: Vec<bool> = sets.iter().flat_map(|s| s.samples.iter().filter_map(move |x| gold_ref.get(&(x.sentence, x.slot)).map(|g| *g == s.synset))).collect();
    let tried_gold: usize = picked.iter().flat_map(|(syn, members)| members.iter().flat_map(|m| occurrences(*syn, m).into_iter().take(per_member)).filter(|o| gold.contains_key(&(o.sentence, o.slot))).map(move |_| 1)).sum();
    let base_rate: Vec<bool> = picked.iter().flat_map(|(syn, members)| members.iter().flat_map(|m| occurrences(*syn, m).into_iter().take(per_member)).filter_map(|o| gold.get(&(o.sentence, o.slot)).map(|g| *g == *syn)).collect::<Vec<_>>()).collect();
    println!(
        "precision on SemCor: {:.1}% of {} shared samples have the set's sense (vs {:.1}% of all {} tried uses — the rate with no filter)",
        100.0 * judged.iter().filter(|x| **x).count() as f64 / judged.len().max(1) as f64, judged.len(),
        100.0 * base_rate.iter().filter(|x| **x).count() as f64 / base_rate.len().max(1) as f64, tried_gold
    );

    std::fs::create_dir_all("data/synonyms").unwrap();
    let mut out = std::io::BufWriter::new(std::fs::File::create("data/synonyms/shared.jsonl").unwrap());
    for s in &sets {
        let samples: Vec<serde_json::Value> = s.samples.iter().map(|x| serde_json::json!({ "text": corpus[x.sentence].join(" "), "slot": x.slot, "fits": x.fits })).collect();
        writeln!(out, "{}", serde_json::json!({ "synset": e.lexicon.synsets[s.synset as usize].id, "definition": e.lexicon.synsets[s.synset as usize].definition, "members": s.members, "samples": samples })).unwrap();
    }
    for s in sets.iter().filter(|s| s.samples.len() >= 2).take(6) {
        println!("\n{} — {}", s.members.join(", "), e.lexicon.synsets[s.synset as usize].definition);
        for x in s.samples.iter().take(3) {
            let mut words = corpus[x.sentence].clone();
            words[x.slot] = format!("[{}]", words[x.slot]);
            println!("   {}   fits: {}", words.join(" "), x.fits.join(", "));
        }
    }
}
