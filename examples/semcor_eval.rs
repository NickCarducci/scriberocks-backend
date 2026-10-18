//! Sense choice on natural, sense-tagged text.
//!
//!   cargo run --release --example semcor_eval [eval-sentences] [train-sentences]
//!
//! SemCor documents are split even/odd: odd documents are evaluated, even
//! ones supply extra sense examples. The Raganato "ALL" set (Senseval/SemEval)
//! is evaluated too. Each set is scored with the shipped sense index (dictionary
//! + Wiktionary examples) and with SemCor examples added, at several margins.
//!
//! Note: WordNet's sense order comes from SemCor counts, so first-listed is
//! unusually strong on SemCor itself; ALL is the fairer comparison.
use scribe::engine::choose_sense;
use scribe::kgram::{self, Example, KgramSenses};
use scribe::semcor::{self, TaggedSentence};
use scribe::usage::sense_head::{self, Candidate, SenseHead};
use scribe::usage::senses::SenseIndex;
use scribe::{Engine, Lexicon, Pos};
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

const WSD: &str = "data/semcor/wsdeval/WSD_Evaluation_Framework";
const MARGINS: &[f32] = &[0.0, 0.05, 0.1, 0.2, 0.3, 0.5];

/// One evaluated occurrence: its candidate senses (first-listed first), the
/// gold senses and the masked-context vector.
struct Case {
    senses: Vec<u32>,
    gold: Vec<u32>,
    vector: Vec<f32>,
    lemma: String,
    pos: Pos,
    /// Index of its sentence in the evaluated list (for sentence context).
    sentence: usize,
    features: Vec<String>,
    /// Lower-cased window around the word (fits the usage model) and the word's index in it.
    window: Vec<String>,
    at: usize,
}

fn load(name: &str, dir: &str) -> Vec<TaggedSentence> {
    let base = format!("{WSD}/{dir}/{name}");
    semcor::parse(&std::fs::read_to_string(format!("{base}.data.xml")).unwrap(), &std::fs::read_to_string(format!("{base}.gold.key.txt")).unwrap())
}

/// Masked-context vectors for each instance, one sentence at a time (the
/// model chunks internally, so memory stays bounded).
fn vectors(e: &Engine, s: &TaggedSentence) -> Vec<Option<Vec<f32>>> {
    let lower: Vec<String> = s.words.iter().map(|w| w.to_lowercase()).collect();
    let masked: Vec<Vec<String>> = s.instances.iter().map(|i| { let mut m = lower.clone(); m[i.index] = "[MASK]".into(); m }).collect();
    match e.usage.as_ref().unwrap().contextual_batch(&masked) {
        Ok(batch) => s.instances.iter().zip(batch).map(|(i, h)| h.get(i.index).cloned()).collect(),
        Err(_) => vec![None; s.instances.len()],
    }
}

fn cases(e: &Engine, keys: &HashMap<String, u32>, sentences: &[TaggedSentence]) -> (Vec<Case>, usize) {
    let (mut out, mut skipped) = (Vec::new(), 0);
    for (n, s) in sentences.iter().enumerate() {
        if n % 500 == 0 { eprintln!("  {n}/{}", sentences.len()) }
        let lower: Vec<String> = s.words.iter().map(|w| w.to_lowercase()).collect();
        for (inst, v) in s.instances.iter().zip(vectors(e, s)) {
            let senses = e.lexicon.senses(&inst.lemma, inst.pos).to_vec();
            let gold: Vec<u32> = inst.keys.iter().filter_map(|k| keys.get(k).copied()).filter(|g| senses.contains(g)).collect();
            match v {
                Some(vector) if !gold.is_empty() && senses.len() > 1 => {
                    let lo = inst.index.saturating_sub(20);
                    let window = lower[lo..(lo + 40).min(lower.len())].to_vec();
                    out.push(Case { senses, gold, vector, lemma: inst.lemma.clone(), pos: inst.pos, sentence: n, features: kgram::features(&lower, inst.index), window, at: inst.index - lo })
                }
                _ => skipped += 1,
            }
        }
    }
    (out, skipped)
}

fn report(label: &str, cases: &[Case], index: &SenseIndex, lex: &Lexicon) {
    let n = cases.len().max(1) as f64;
    let first = cases.iter().filter(|c| c.gold.contains(&c.senses[0])).count();
    println!("{label}: {} ambiguous words; first-listed {:.1}%", cases.len(), 100.0 * first as f64 / n);
    for &m in MARGINS {
        let (mut right, mut overrides) = (0, 0);
        for c in cases {
            let (s, _) = choose_sense(&c.senses, |s| index.score(s, &c.vector, lex), m).unwrap();
            right += c.gold.contains(&s) as usize;
            overrides += (s != c.senses[0]) as usize;
        }
        println!("  margin {m:<4}: {:.1}% correct, overrides first-listed {:.1}%", 100.0 * right as f64 / n, 100.0 * overrides as f64 / n);
    }
}

fn report_kgram(label: &str, cases: &[Case], head: &KgramSenses, index: &SenseIndex, lex: &Lexicon) {
    let n = cases.len().max(1) as f64;
    let known = cases.iter().filter(|c| head.knows(&c.lemma, c.pos)).count();
    println!("{label}: k-gram head (lemma seen in training for {:.1}%)", 100.0 * known as f64 / n);
    for m in [2.0f32, 4.0, 6.0, 8.0, 12.0, 16.0] {
        let (mut alone, mut combined, mut overrides) = (0, 0, 0);
        for c in cases {
            let k = head.choose(&c.lemma, c.pos, &c.senses, &c.features, m).unwrap();
            alone += c.gold.contains(&k) as usize;
            overrides += (k != c.senses[0]) as usize;
            let s = match head.knows(&c.lemma, c.pos) {
                true => k,
                false => choose_sense(&c.senses, |s| index.score(s, &c.vector, lex), 0.3).unwrap().0,
            };
            combined += c.gold.contains(&s) as usize;
        }
        println!("  margin {m:<3}: {:.1}% (overrides {:.1}%); with usage vectors for unseen lemmas {:.1}%", 100.0 * alone as f64 / n, 100.0 * overrides as f64 / n, 100.0 * combined as f64 / n);
    }
}

fn candidates(c: &Case, lex: &Lexicon) -> Vec<Candidate> {
    c.senses.iter().enumerate().map(|(rank, &s)| Candidate { synset: s, category: lex.synsets[s as usize].category, rank }).collect()
}

/// Supersenses of the other cases in each case's sentence, given one chosen
/// synset per case.
fn contexts(cases: &[Case], chosen: &[u32], lex: &Lexicon) -> Vec<Vec<u16>> {
    let mut by_sentence: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, c) in cases.iter().enumerate() {
        by_sentence.entry(c.sentence).or_default().push(i);
    }
    (0..cases.len())
        .map(|i| by_sentence[&cases[i].sentence].iter().filter(|&&j| j != i).map(|&j| lex.synsets[chosen[j] as usize].category).collect())
        .collect()
}

fn head_scores(cases: &[Case], head: &SenseHead, context: Option<&[Vec<u16>]>, lex: &Lexicon) -> Vec<Vec<f32>> {
    cases.iter().enumerate().map(|(i, c)| head.scores_in_context(&c.vector, &candidates(c, lex), context.map_or(&[][..], |x| &x[i])).unwrap()).collect()
}

/// Index chosen per case: the best candidate if it beats first-listed by `m`.
fn pick(scores: &[Vec<f32>], m: f32) -> Vec<usize> {
    scores.iter().map(|s| { let best = (0..s.len()).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap_or(0); if s[best] - s[0] > m { best } else { 0 } }).collect()
}

fn sweep(label: &str, cases: &[Case], scores: &[Vec<f32>]) {
    let n = cases.len().max(1) as f64;
    for m in [0.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0] {
        let k = pick(scores, m);
        let right = cases.iter().zip(&k).filter(|(c, k)| c.gold.contains(&c.senses[**k])).count();
        let overrides = k.iter().filter(|&&k| k != 0).count();
        println!("{label} margin {m:<3}: {:.1}% (overrides first-listed {:.1}%)", 100.0 * right as f64 / n, 100.0 * overrides as f64 / n);
    }
}

fn main() {
    let arg = |i: usize, d: usize| std::env::args().nth(i).and_then(|a| a.parse().ok()).unwrap_or(d);
    let (eval_cap, train_cap) = (arg(1, 2000), arg(2, 8000));
    // USAGE=usage-sense evaluates another usage model (and its sense index).
    let usage = std::env::var("USAGE").unwrap_or_else(|_| "usage".into());
    let mut e = Engine::load(Path::new("models")).unwrap();
    if usage != "usage" {
        let dir = Path::new("models").join(&usage);
        e.usage = Some(scribe::usage::model::UsageModel::load(&dir, scribe::usage::model::Device::Cpu).unwrap());
        let mut index: SenseIndex = postcard::from_bytes(&std::fs::read(dir.join("senses.bin")).unwrap()).unwrap();
        index.build_backoff(&e.lexicon);
        e.senses = Some(index);
        eprintln!("usage model: models/{usage}");
    }
    let keys = semcor::sense_keys(Path::new("data/oewn"), &e.lexicon).unwrap();

    let semcor = load("semcor", "Training_Corpora/SemCor");
    let docs: Vec<&str> = semcor.iter().map(|s| s.document.as_str()).collect::<BTreeSet<_>>().into_iter().collect();
    let odd: BTreeSet<&str> = docs.iter().skip(1).step_by(2).copied().collect();
    let (held, train): (Vec<_>, Vec<_>) = semcor.iter().cloned().partition(|s| odd.contains(s.document.as_str()));
    let every = |v: &[TaggedSentence], cap: usize| -> Vec<TaggedSentence> {
        let step = (v.len() / cap.max(1)).max(1);
        v.iter().step_by(step).take(cap).cloned().collect()
    };
    let all_keys: usize = semcor.iter().flat_map(|s| &s.instances).count();
    let mapped: usize = semcor.iter().flat_map(|s| &s.instances).filter(|i| i.keys.iter().any(|k| keys.contains_key(k))).count();
    eprintln!("SemCor: {} sentences in {} documents; {mapped}/{all_keys} sense keys map to the lexicon", semcor.len(), docs.len());

    eprintln!("vectors: SemCor held-out documents");
    let (held_cases, skipped) = cases(&e, &keys, &every(&held, eval_cap));
    eprintln!("  ({skipped} unambiguous or unmapped skipped)");
    eprintln!("vectors: ALL (Senseval/SemEval)");
    let (all_cases, _) = cases(&e, &keys, &every(&load("ALL", "Evaluation_Datasets/ALL"), eval_cap));

    let shipped = e.senses.as_ref().unwrap();
    report("SemCor held-out, shipped index", &held_cases, shipped, &e.lexicon);
    report("ALL, shipped index", &all_cases, shipped, &e.lexicon);

    eprintln!("vectors: SemCor training documents");
    let mut index: SenseIndex = postcard::from_bytes(&std::fs::read(Path::new("models").join(&usage).join("senses.bin")).unwrap()).unwrap();
    let (train_cases, _) = cases(&e, &keys, &every(&train, train_cap));
    for c in &train_cases {
        index.add(c.gold[0], &c.vector);
    }
    let examples: Vec<Example> = train_cases.iter().map(|c| Example { lemma: c.lemma.clone(), pos: c.pos, sense: c.gold[0], features: c.features.clone() }).collect();
    let head = KgramSenses::train(&examples, 5);
    index.build_backoff(&e.lexicon);
    println!("(+ {} SemCor examples from {} training sentences)", train_cases.len(), train_cap.min(train.len()));
    report("SemCor held-out, + SemCor examples", &held_cases, &index, &e.lexicon);
    report("ALL, + SemCor examples", &all_cases, &index, &e.lexicon);
    report_kgram("SemCor held-out", &held_cases, &head, &index, &e.lexicon);
    report_kgram("ALL", &all_cases, &head, &index, &e.lexicon);

    let examples = |context: &dyn Fn(usize) -> Vec<u16>| -> Vec<sense_head::Example> {
        train_cases
            .iter()
            .enumerate()
            .map(|(i, c)| sense_head::Example { vector: c.vector.clone(), candidates: candidates(c, &e.lexicon), gold: c.senses.iter().position(|s| *s == c.gold[0]).unwrap(), context: context(i) })
            .collect()
    };
    let opts = sense_head::TrainOptions::default();
    let cats = e.lexicon.categories.len();
    let plain = SenseHead::train(&examples(&|_| vec![]), cats, &opts, scribe::usage::model::Device::Cpu).unwrap();
    // Context head: trained with the other words' gold senses.
    let gold_chosen: Vec<u32> = train_cases.iter().map(|c| c.gold[0]).collect();
    let train_context = contexts(&train_cases, &gold_chosen, &e.lexicon);
    let with_context = SenseHead::train(&examples(&|i| train_context[i].clone()), cats, &opts, scribe::usage::model::Device::Cpu).unwrap();
    for (label, cases) in [("SemCor held-out", &held_cases), ("ALL", &all_cases)] {
        let first = head_scores(cases, &plain, None, &e.lexicon);
        sweep(&format!("{label}, sense head"), cases, &first);
        // First pass at the dev-chosen margin, then rescore with sentence context.
        let chosen: Vec<u32> = cases.iter().zip(pick(&first, 5.0)).map(|(c, k)| c.senses[k]).collect();
        let ctx = contexts(cases, &chosen, &e.lexicon);
        sweep(&format!("{label}, sense head + sentence context"), cases, &head_scores(cases, &with_context, Some(&ctx), &e.lexicon));
    }

    // ── Editor view: senses as synonym options ─────────────────────────────
    let stop = |w: &str| (e.usage.as_ref().unwrap().vocab.id(w) as usize) < scribe::usage::vocab::SPECIALS.len() + 100;
    for (label, cases) in [("SemCor held-out", &held_cases), ("ALL", &all_cases)] {
        let scores = head_scores(cases, &plain, None, &e.lexicon);
        editor_report(label, cases, &scores, &e.lexicon, &stop);
        editor_with_usage(label, cases, &scores, &e.lexicon, &stop, e.usage.as_ref().unwrap());
        editor_hierarchical(label, cases, &scores, &e.lexicon, &stop, e.usage.as_ref().unwrap());
    }
}

/// Other usable single-word members of a synset (the options it offers instead of `lemma`).
fn options(lex: &Lexicon, synset: u32, lemma: &str, stop: &dyn Fn(&str) -> bool) -> Vec<String> {
    lex.synsets[synset as usize]
        .members
        .iter()
        .map(|m| m.to_lowercase())
        .filter(|m| m != lemma && !m.contains(' ') && scribe::usage::synonyms::usable_member(m, stop))
        .collect()
}

/// Groups of eligible senses: "fine" (one sense each), "options" (senses that
/// share an option merge), "options+category" (also merge same supersense).
fn groups(eligible: &[u32], opts: &HashMap<u32, Vec<String>>, lex: &Lexicon, mode: &str) -> Vec<Vec<u32>> {
    let mut parent: Vec<usize> = (0..eligible.len()).collect();
    fn find(p: &mut Vec<usize>, i: usize) -> usize { if p[i] != i { let r = find(p, p[i]); p[i] = r } p[i] }
    for i in 0..eligible.len() {
        for j in i + 1..eligible.len() {
            let share = opts[&eligible[i]].iter().any(|m| opts[&eligible[j]].contains(m));
            let same_cat = lex.synsets[eligible[i] as usize].category == lex.synsets[eligible[j] as usize].category;
            let merge = match mode { "options" => share, "options+category" => share || same_cat, _ => false };
            if merge { let (a, b) = (find(&mut parent, i), find(&mut parent, j)); parent[a] = b; }
        }
    }
    let mut out: HashMap<usize, Vec<u32>> = HashMap::new();
    for i in 0..eligible.len() { let r = find(&mut parent, i); out.entry(r).or_default().push(eligible[i]); }
    let mut g: Vec<Vec<u32>> = out.into_values().collect();
    g.sort();
    g
}

fn editor_report(label: &str, cases: &[Case], scores: &[Vec<f32>], lex: &Lexicon, stop: &dyn Fn(&str) -> bool) {
    for mode in ["fine", "options", "options+category"] {
        // (eligible, suggested, correct) per abstention margin, and first-listed.
        let margins = [0.0f32, 1.0, 2.0, 4.0, 8.0];
        let mut tally = vec![(0usize, 0usize, 0usize); margins.len()];
        let mut first = (0usize, 0usize);
        let mut sizes = (0usize, 0usize);
        let mut gold_has_options = 0usize;
        for (c, s) in cases.iter().zip(scores) {
            let opts: HashMap<u32, Vec<String>> = c.senses.iter().map(|&x| (x, options(lex, x, &c.lemma, stop))).collect();
            let eligible: Vec<u32> = c.senses.iter().copied().filter(|x| !opts[x].is_empty()).collect();
            if eligible.is_empty() { continue }
            let gs = groups(&eligible, &opts, lex, mode);
            sizes.0 += gs.len(); sizes.1 += 1;
            gold_has_options += eligible.iter().any(|x| c.gold.contains(x)) as usize;
            let score_of = |x: u32| c.senses.iter().position(|y| *y == x).map_or(f32::NEG_INFINITY, |k| s[k]);
            // All senses compete; the ones without options form one "no
            // suggestion" group — if it wins, the editor abstains.
            let none: Vec<u32> = c.senses.iter().copied().filter(|x| opts[x].is_empty()).collect();
            let mut ranked: Vec<(f32, Option<&Vec<u32>>)> = gs.iter().map(|g| (g.iter().map(|&x| score_of(x)).fold(f32::NEG_INFINITY, f32::max), Some(g))).collect();
            if !none.is_empty() { ranked.push((none.iter().map(|&x| score_of(x)).fold(f32::NEG_INFINITY, f32::max), None)); }
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
            let margin = if ranked.len() > 1 { ranked[0].0 - ranked[1].0 } else { f32::INFINITY };
            let picked = ranked[0].1;
            let right = picked.is_some_and(|g| g.iter().any(|x| c.gold.contains(x)));
            for (k, &m) in margins.iter().enumerate() {
                tally[k].0 += 1;
                if picked.is_some() && margin >= m { tally[k].1 += 1; tally[k].2 += right as usize; }
            }
            // First-listed eligible sense's group, never abstains.
            let f = gs.iter().find(|g| g.contains(&eligible[0])).unwrap();
            first.0 += 1; first.1 += f.iter().any(|x| c.gold.contains(x)) as usize;
        }
        println!("{label}, editor [{mode}] ({} eligible words, {:.1} meanings with options each; true sense has options in {:.1}%): first-listed {:.1}% precision at 100% coverage",
            first.0, sizes.0 as f64 / sizes.1.max(1) as f64, 100.0 * gold_has_options as f64 / first.0.max(1) as f64, 100.0 * first.1 as f64 / first.0.max(1) as f64);
        for (k, m) in margins.iter().enumerate() {
            let (n, sug, ok) = tally[k];
            println!("  head, abstain below margin {m:<3}: precision {:.1}%, coverage {:.1}%", 100.0 * ok as f64 / sug.max(1) as f64, 100.0 * sug as f64 / n.max(1) as f64);
        }
    }
}

/// Editor view followed by the usage model: each meaning's options are tried
/// in the word's slot, and the meaning's fit (log of the options' mean
/// probability, or of the best option's) is added to the head's score with
/// weight beta, relative to the case's mean fit. beta 0 = the editor view.
fn editor_with_usage(label: &str, cases: &[Case], scores: &[Vec<f32>], lex: &Lexicon, stop: &dyn Fn(&str) -> bool, usage: &scribe::usage::model::UsageModel) {
    let margins = [0.0f32, 1.0, 2.0, 4.0, 8.0];
    let betas = [0.0f32, 0.25, 0.5, 1.0, 2.0];
    // Per case: groups ("options" mode) with head score, mean-fit and max-fit; plus the no-suggestion score.
    struct G { head: f32, mean_fit: f32, max_fit: f32, right: bool }
    let mut prepared: Vec<(Vec<G>, Option<f32>)> = Vec::new();
    for (n, (c, s)) in cases.iter().zip(scores).enumerate() {
        if n % 500 == 0 { eprintln!("  usage fit {n}/{}", cases.len()) }
        let opts: HashMap<u32, Vec<String>> = c.senses.iter().map(|&x| (x, options(lex, x, &c.lemma, stop))).collect();
        let eligible: Vec<u32> = c.senses.iter().copied().filter(|x| !opts[x].is_empty()).collect();
        if eligible.is_empty() { continue }
        let gs = groups(&eligible, &opts, lex, "options");
        let all: Vec<String> = eligible.iter().flat_map(|x| opts[x].clone()).collect::<BTreeSet<_>>().into_iter().collect();
        let lp: HashMap<String, f32> = all.iter().cloned().zip(usage.candidate_scores(&c.window, c.at, &all).unwrap()).collect();
        let score_of = |x: u32| c.senses.iter().position(|y| *y == x).map_or(f32::NEG_INFINITY, |k| s[k]);
        let out: Vec<G> = gs.iter().map(|g| {
            let o: BTreeSet<&String> = g.iter().flat_map(|x| &opts[x]).collect();
            let v: Vec<f32> = o.iter().map(|w| lp[*w].max(-30.0)).collect();
            let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean = mx + (v.iter().map(|x| (x - mx).exp()).sum::<f32>() / v.len() as f32).ln();
            G { head: g.iter().map(|&x| score_of(x)).fold(f32::NEG_INFINITY, f32::max), mean_fit: mean, max_fit: mx, right: g.iter().any(|x| c.gold.contains(x)) }
        }).collect();
        let none: Vec<u32> = c.senses.iter().copied().filter(|x| opts[x].is_empty()).collect();
        let none_score = (!none.is_empty()).then(|| none.iter().map(|&x| score_of(x)).fold(f32::NEG_INFINITY, f32::max));
        prepared.push((out, none_score));
    }
    for fit in ["mean", "max"] {
        for &beta in &betas {
            let mut tally = vec![(0usize, 0usize, 0usize); margins.len()];
            let mut always = (0usize, 0usize);
            for (gs, none) in &prepared {
                let f = |g: &G| if fit == "mean" { g.mean_fit } else { g.max_fit };
                let avg = gs.iter().map(f).sum::<f32>() / gs.len() as f32;
                let mut ranked: Vec<(f32, Option<bool>)> = gs.iter().map(|g| (g.head + beta * (f(g) - avg), Some(g.right))).collect();
                // Best option-bearing group, never abstaining (coverage 100%).
                let best = ranked.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
                always.0 += 1; always.1 += best.1.unwrap() as usize;
                if let Some(x) = none { ranked.push((*x, None)) }
                ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
                let margin = if ranked.len() > 1 { ranked[0].0 - ranked[1].0 } else { f32::INFINITY };
                for (k, &m) in margins.iter().enumerate() {
                    tally[k].0 += 1;
                    if let Some(r) = ranked[0].1 { if margin >= m { tally[k].1 += 1; tally[k].2 += r as usize } }
                }
            }
            let cells: Vec<String> = margins.iter().zip(&tally).map(|(m, (n, sug, ok))| format!("m{m}: {:.1}%@{:.1}%", 100.0 * *ok as f64 / (*sug).max(1) as f64, 100.0 * *sug as f64 / (*n).max(1) as f64)).collect();
            println!("{label}, editor + usage fit [{fit}, beta {beta}]: never-abstain {:.1}% | precision@coverage {}", 100.0 * always.1 as f64 / always.0.max(1) as f64, cells.join("  "));
        }
    }
}

/// Two-step editor view: first the broad category (supersense), scored by the
/// head's best sense in it plus gamma x the usage model's supersense
/// log-probability for the masked word; then, within that category only, the
/// option groups (meanings sharing a synonym), picked by the head.
/// Reports the category step's precision, the final (option-group) precision,
/// and how often the chosen category holds a single option group (the
/// category alone settles the suggestion).
fn editor_hierarchical(label: &str, cases: &[Case], scores: &[Vec<f32>], lex: &Lexicon, stop: &dyn Fn(&str) -> bool, usage: &scribe::usage::model::UsageModel) {
    let margins = [0.0f32, 1.0, 2.0, 4.0, 8.0];
    let gammas = [0.0f32, 0.5, 1.0, 2.0, 4.0];
    // Per case: (category, head score, right category, option groups [(head, right)]); none-group (head, best lp).
    struct Cat { lp: f32, head: f32, right: bool, groups: Vec<(f32, bool)>, whole: Vec<(f32, bool)> }
    let mut prepared: Vec<(Vec<Cat>, Option<(f32, f32)>)> = Vec::new();
    for (n, (c, s)) in cases.iter().zip(scores).enumerate() {
        if n % 1000 == 0 { eprintln!("  supersenses {n}/{}", cases.len()) }
        let opts: HashMap<u32, Vec<String>> = c.senses.iter().map(|&x| (x, options(lex, x, &c.lemma, stop))).collect();
        let eligible: Vec<u32> = c.senses.iter().copied().filter(|x| !opts[x].is_empty()).collect();
        if eligible.is_empty() { continue }
        let mut masked = c.window.clone();
        masked[c.at] = "[MASK]".into();
        let lp = usage.supersense_log_probs(&masked).unwrap().and_then(|v| v.get(c.at).cloned()).expect("usage-sense model with supersenses");
        let score_of = |x: u32| c.senses.iter().position(|y| *y == x).map_or(f32::NEG_INFINITY, |k| s[k]);
        let cat_of = |x: u32| lex.synsets[x as usize].category;
        let cats: BTreeSet<u16> = eligible.iter().map(|&x| cat_of(x)).collect();
        // Synonym groups over all eligible meanings (not split by category).
        let all_groups = groups(&eligible, &opts, lex, "options");
        let out: Vec<Cat> = cats.iter().map(|&k| {
            let whole = all_groups.iter().filter(|g| g.iter().any(|&x| cat_of(x) == k))
                .map(|g| (g.iter().map(|&x| score_of(x)).fold(f32::NEG_INFINITY, f32::max), g.iter().any(|x| c.gold.contains(x)))).collect::<Vec<_>>();
            let members: Vec<u32> = eligible.iter().copied().filter(|&x| cat_of(x) == k).collect();
            let groups = groups(&members, &opts, lex, "options").into_iter()
                .map(|g| (g.iter().map(|&x| score_of(x)).fold(f32::NEG_INFINITY, f32::max), g.iter().any(|x| c.gold.contains(x)))).collect::<Vec<_>>();
            Cat { whole, lp: lp[k as usize], head: members.iter().map(|&x| score_of(x)).fold(f32::NEG_INFINITY, f32::max), right: members.iter().any(|x| c.gold.contains(x)), groups }
        }).collect();
        let none: Vec<u32> = c.senses.iter().copied().filter(|x| opts[x].is_empty()).collect();
        let none = (!none.is_empty()).then(|| (none.iter().map(|&x| score_of(x)).fold(f32::NEG_INFINITY, f32::max), none.iter().map(|&x| lp[cat_of(x) as usize]).fold(f32::NEG_INFINITY, f32::max)));
        prepared.push((out, none));
    }
    for &gamma in &gammas {
        let mut tally = vec![(0usize, 0usize, 0usize, 0usize, 0usize); margins.len()]; // n, suggested, category right, final right, single-group
        let mut whole_right = vec![0usize; margins.len()];
        for (cats, none) in &prepared {
            let mut ranked: Vec<(f32, Option<&Cat>)> = cats.iter().map(|k| (k.head + gamma * k.lp, Some(k))).collect();
            if let Some((h, l)) = none { ranked.push((h + gamma * l, None)) }
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
            let margin = if ranked.len() > 1 { ranked[0].0 - ranked[1].0 } else { f32::INFINITY };
            for (i, &m) in margins.iter().enumerate() {
                tally[i].0 += 1;
                if let Some(k) = ranked[0].1 {
                    if margin >= m {
                        let g = k.groups.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
                        whole_right[i] += k.whole.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1 as usize;
                        tally[i].1 += 1; tally[i].2 += k.right as usize; tally[i].3 += g.1 as usize; tally[i].4 += (k.groups.len() == 1) as usize;
                    }
                }
            }
        }
        let cells: Vec<String> = margins.iter().zip(&tally).zip(&whole_right).map(|((m, (n, sug, cat, fin, one)), w)| {
            let p = |x: usize| 100.0 * x as f64 / (*sug).max(1) as f64;
            format!("m{m}: category {:.1}% final {:.1}% whole-groups {:.1}% @{:.1}% (settled by category {:.0}%)", p(*cat), p(*fin), p(*w), 100.0 * *sug as f64 / (*n).max(1) as f64, p(*one))
        }).collect();
        println!("{label}, editor two-step [gamma {gamma}]: {}", cells.join(" | "));
    }
}
