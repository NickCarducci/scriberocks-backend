//! Shared pieces for training ScribeSLM's language model (examples/slm_build.rs
//! from scratch, examples/slm_adapt.rs from BERT-mini): BERT-style masking
//! (whole words), the masked-word loss, and the joint dictionary objective —
//! in SemCor, a hidden word's slot should accept the word or its labelled
//! meaning's synonyms.
use crate::bert::Bert;
use crate::{semcor, Lexicon};
use candle_core::{Device, Result, Tensor};
use std::path::Path;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    pub fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
}

/// Batches of sentence indices, by length (shortest first), each holding at
/// most `max_sentences` and at most `max_pieces` padded pieces (sentences ×
/// longest length in the batch). Bounds memory and time per step: a fixed
/// sentence count lets batches of the longest sentences cost ~5× the rest.
pub fn token_batches(lengths: &[usize], max_sentences: usize, max_pieces: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&i| lengths[i]);
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for i in order {
        // Sorted, so this sentence is the longest so far in the batch.
        let padded = (current.len() + 1) * lengths[i];
        if !current.is_empty() && (current.len() == max_sentences || padded > max_pieces) {
            out.push(std::mem::take(&mut current));
        }
        current.push(i);
    }
    if !current.is_empty() { out.push(current) }
    out
}

/// A training sentence: piece ids ([CLS] … [SEP]) and each word's first piece.
pub struct Encoded {
    pub ids: Vec<u32>,
    pub first: Vec<u16>,
}

/// Whole-word masking: ~15% of words, all of a word's pieces together; per
/// word 80% [MASK], 10% random pieces, 10% unchanged. Returns input rows,
/// masked (row, column) positions and the original ids there.
pub fn mask_whole_words(bert: &Bert, batch: &[&Encoded], rng: &mut Rng) -> (Vec<Vec<u32>>, Vec<(usize, usize)>, Vec<u32>) {
    let m = bert.special("[MASK]");
    let (mut rows, mut at, mut gold) = (Vec::new(), Vec::new(), Vec::new());
    for (r, e) in batch.iter().enumerate() {
        let mut row = e.ids.clone();
        let end = row.len().saturating_sub(1); // [SEP]
        for (k, &f) in e.first.iter().enumerate() {
            let (start, stop) = (f as usize, e.first.get(k + 1).map_or(end, |&x| x as usize).min(end));
            if start >= stop || rng.unit() >= 0.15 { continue }
            let x = rng.unit();
            for c in start..stop {
                at.push((r, c));
                gold.push(row[c]);
                if x < 0.8 { row[c] = m } else if x < 0.9 { row[c] = 5 + (rng.next() % (bert.vocab_len() as u64 - 5)) as u32 }
            }
        }
        rows.push(row);
    }
    (rows, at, gold)
}

/// Hidden positions are padded up to a multiple of this (zero-weight dummies),
/// so the [positions × vocabulary] prediction table comes in few sizes: candle's
/// Metal backend reuses buffers only of an exact size, and a new size every
/// step (each batch hides a different number of words) piled up ~10 GB.
pub const POSITION_BUCKET: usize = 128;

/// Mean negative log-probability of the original pieces at the masked positions.
pub fn masked_loss(bert: &Bert, rows: &[Vec<u32>], at: &[(usize, usize)], gold: &[u32], dev: &Device) -> Result<Tensor> {
    let n = at.len();
    let padded = n.div_ceil(POSITION_BUCKET) * POSITION_BUCKET;
    let mut at_p = at.to_vec();
    at_p.resize(padded, (0, 0));
    let mut gold_p = gold.to_vec();
    gold_p.resize(padded, 0);
    let weight: Vec<f32> = (0..padded).map(|i| if i < n { 1.0 / n as f32 } else { 0.0 }).collect();
    let lp = bert.masked_log_probs(rows, &at_p)?;
    let picked = lp.gather(&Tensor::new(gold_p.as_slice(), dev)?.unsqueeze(1)?, 1)?.squeeze(1)?;
    (picked * Tensor::from_vec(weight, padded, dev)?)?.sum_all()?.neg()
}

/// A SemCor word (in base form) whose hidden slot should accept `accept`.
pub struct DictExample {
    pub ids: Vec<u32>,
    pub at: usize,
    pub accept: Vec<u32>,
}

/// SemCor words in base form whose labelled meaning has at least one other
/// single-piece synonym (so the objective says something masking alone doesn't).
pub fn dictionary_examples(bert: &Bert, lex: &Lexicon, semcor_base: &str, oewn: &Path) -> std::result::Result<Vec<DictExample>, Box<dyn std::error::Error>> {
    let sentences = semcor::parse(&std::fs::read_to_string(format!("{semcor_base}.data.xml"))?, &std::fs::read_to_string(format!("{semcor_base}.gold.key.txt"))?);
    let keys = semcor::sense_keys(oewn, lex)?;
    let mut out = Vec::new();
    for s in &sentences {
        let (ids, first) = bert.encode_words(&s.words);
        for inst in &s.instances {
            let word = s.words[inst.index].to_lowercase();
            let Some(&gold) = inst.keys.iter().find_map(|k| keys.get(k)) else { continue };
            let (Some(id), Some(&at)) = (bert.single_piece(&word), first.get(inst.index)) else { continue };
            if word != inst.lemma.to_lowercase() || at >= ids.len() { continue }
            let mut accept = vec![id];
            for m in &lex.synsets[gold as usize].members {
                if let Some(x) = bert.single_piece(m).filter(|_| !m.contains(' ')) {
                    if !accept.contains(&x) { accept.push(x) }
                }
            }
            if accept.len() > 1 { out.push(DictExample { ids: ids.clone(), at, accept }) }
        }
    }
    Ok(out)
}

/// A random dictionary batch of whole sentences: up to `max_examples`, and (when
/// `max_pieces` > 0) at most `max_pieces` padded pieces, so long SemCor sentences
/// come in smaller batches rather than being cropped.
pub fn dictionary_batch<'a>(pool: &'a [DictExample], max_examples: usize, max_pieces: usize, rng: &mut Rng) -> Vec<&'a DictExample> {
    let mut out: Vec<&DictExample> = Vec::new();
    let mut longest = 0;
    while out.len() < max_examples {
        let e = &pool[(rng.next() % pool.len() as u64) as usize];
        let l = longest.max(e.ids.len());
        if max_pieces > 0 && !out.is_empty() && (out.len() + 1) * l > max_pieces { break }
        longest = l;
        out.push(e);
    }
    out
}

/// −log of the probability mass each hidden slot puts on its acceptable words.
pub fn dictionary_loss(bert: &Bert, batch: &[&DictExample], dev: &Device) -> Result<Tensor> {
    let m = bert.special("[MASK]");
    let rows: Vec<Vec<u32>> = batch.iter().map(|e| { let mut r = e.ids.clone(); r[e.at] = m; r }).collect();
    let at: Vec<(usize, usize)> = batch.iter().enumerate().map(|(i, e)| (i, e.at)).collect();
    let lp = bert.masked_log_probs(&rows, &at)?;
    let k = batch.iter().map(|e| e.accept.len()).max().unwrap_or(1);
    let (mut idx, mut pad) = (Vec::new(), Vec::new());
    for e in batch {
        for j in 0..k {
            idx.push(*e.accept.get(j).unwrap_or(&e.accept[0]));
            pad.push(if j < e.accept.len() { 0f32 } else { -1e9 });
        }
    }
    let n = batch.len();
    let picked = (lp.gather(&Tensor::from_vec(idx, (n, k), dev)?, 1)? + Tensor::from_vec(pad, (n, k), dev)?)?;
    let max = picked.max_keepdim(1)?;
    let lse = (picked.broadcast_sub(&max)?.exp()?.sum_keepdim(1)?.log()? + max)?;
    lse.neg()?.mean_all()
}

#[cfg(test)]
mod batching_tests {
    use super::*;

    #[test]
    fn token_batches_cover_every_sentence_once_within_both_limits() {
        let lengths: Vec<usize> = (0..1000).map(|i| 5 + (i * 37) % 124).collect();
        let batches = token_batches(&lengths, 128, 4096);
        let mut seen: Vec<usize> = batches.concat();
        seen.sort();
        assert_eq!(seen, (0..1000).collect::<Vec<_>>());
        for b in &batches {
            let longest = b.iter().map(|&i| lengths[i]).max().unwrap();
            assert!(b.len() <= 128 && b.len() * longest <= 4096, "{} × {longest}", b.len());
        }
        // Short sentences still fill whole batches.
        assert_eq!(token_batches(&[10; 300], 128, 4096)[0].len(), 128);
    }

    /// Padding the hidden positions must not change the loss.
    #[test]
    fn padded_masked_loss_equals_the_plain_mean() {
        let dir = Path::new("data/bert-mini");
        if !dir.join("model.safetensors").exists() { return }
        let bert = Bert::load(dir, Device::Cpu).unwrap();
        let words: Vec<String> = "the cat sat on the mat and the dog sat on a rug".split(' ').map(String::from).collect();
        let rows = vec![bert.encode_sentence(&words)];
        let at: Vec<(usize, usize)> = vec![(0, 2), (0, 5), (0, 9)];
        let gold: Vec<u32> = at.iter().map(|&(r, c)| rows[r][c]).collect();
        let lp = bert.masked_log_probs(&rows, &at).unwrap().to_vec2::<f32>().unwrap();
        let plain = -(0..3).map(|i| lp[i][gold[i] as usize]).sum::<f32>() / 3.0;
        let padded = masked_loss(&bert, &rows, &at, &gold, &Device::Cpu).unwrap().to_scalar::<f32>().unwrap();
        assert!((plain - padded).abs() < 1e-4, "{plain} vs {padded}");
    }

    #[test]
    fn dictionary_batches_keep_whole_sentences_within_the_piece_budget() {
        let pool: Vec<DictExample> = (0..50).map(|i| DictExample { ids: vec![1; if i % 5 == 0 { 120 } else { 20 }], at: 1, accept: vec![1, 2] }).collect();
        let mut rng = Rng(9);
        for _ in 0..200 {
            let b = dictionary_batch(&pool, 64, 2048, &mut rng);
            let longest = b.iter().map(|e| e.ids.len()).max().unwrap();
            assert!(!b.is_empty() && b.len() <= 64 && (b.len() == 1 || b.len() * longest <= 2048));
            assert!(b.iter().all(|e| e.ids.len() == 120 || e.ids.len() == 20), "never cropped");
        }
        assert_eq!(dictionary_batch(&pool, 64, 0, &mut rng).len(), 64, "no budget: fixed size");
    }

    #[test]
    fn a_sentence_longer_than_the_budget_gets_a_batch_of_its_own() {
        let b = token_batches(&[5, 5000, 6], 128, 4096);
        assert!(b.iter().any(|x| x == &vec![1]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny model must memorize a handful of sentences (the standard
    /// "can it overfit?" bug check), in both layouts.
    fn overfits(pre_ln: bool) -> f32 {
        use candle_nn::{AdamW, Optimizer, ParamsAdamW};
        let lines = ["the cat sat on the mat .", "a dog ran in the park .", "she drank hot tea today .", "we built a small boat .",
                     "the old man read a book .", "birds sang in the tall tree .", "he painted the red door .", "they sold fresh bread ."];
        let sents: Vec<Vec<String>> = lines.iter().map(|l| l.split(' ').map(String::from).collect()).collect();
        let mut vocab: Vec<String> = crate::wordpiece::SPECIALS.iter().map(|s| s.to_string()).collect();
        for w in sents.iter().flatten() { if !vocab.contains(w) { vocab.push(w.clone()) } }
        let dir = std::env::temp_dir().join(format!("scribe-overfit-{pre_ln}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("vocab.txt"), vocab.join("\n")).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::json!({"hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
            "intermediate_size": 64, "max_position_embeddings": 16, "layer_norm_eps": 1e-12, "pre_layer_norm": pre_ln}).to_string()).unwrap();
        let (bert, vars) = Bert::new_random(&dir, Device::Cpu).unwrap();
        let data: Vec<Encoded> = sents.iter().map(|s| { let (ids, first) = bert.encode_words(s); Encoded { ids, first: first.iter().map(|&f| f as u16).collect() } }).collect();
        let mut opt = AdamW::new(vars.all_vars(), ParamsAdamW { lr: 3e-3, weight_decay: 0.0, ..Default::default() }).unwrap();
        let mut rng = Rng(5);
        let refs: Vec<&Encoded> = data.iter().collect();
        let mut last = 0.0;
        for step in 0..600 {
            let (rows, at, gold) = mask_whole_words(&bert, &refs, &mut rng);
            if at.is_empty() { continue }
            let loss = masked_loss(&bert, &rows, &at, &gold, &Device::Cpu).unwrap();
            if step >= 550 { last += loss.to_scalar::<f32>().unwrap() / 50.0 }
            opt.backward_step(&loss).unwrap();
        }
        last
    }

    #[test]
    fn a_tiny_model_memorizes_a_few_sentences() {
        let (post, pre) = (overfits(false), overfits(true));
        assert!(pre < 1.0 && post < 1.0, "loss after 600 steps: pre-LN {pre:.3}, post-LN {post:.3} (unigram ≈ 3.3)");
    }

    #[test]
    fn whole_word_masking_masks_all_pieces_of_a_word_and_never_specials() {
        let dir = Path::new("data/bert-mini");
        if !dir.join("model.safetensors").exists() { return }
        let bert = Bert::load(dir, Device::Cpu).unwrap();
        let words: Vec<String> = "the unaffable cat sat on a mat".split(' ').map(String::from).collect();
        let (ids, first) = bert.encode_words(&words);
        let e = Encoded { ids: ids.clone(), first: first.iter().map(|&f| f as u16).collect() };
        let mut rng = Rng(3);
        let (mut masked, mut total) = (0, 0);
        for _ in 0..400 {
            let (_, at, gold) = mask_whole_words(&bert, &[&e], &mut rng);
            assert!(at.iter().all(|&(_, c)| c != 0 && c != ids.len() - 1), "never [CLS]/[SEP]");
            // "unaffable" is several pieces: masked together or not at all.
            let span: Vec<usize> = (first[1]..first[2]).collect();
            let hit = span.iter().filter(|c| at.iter().any(|&(_, x)| x == **c)).count();
            assert!(hit == 0 || hit == span.len());
            assert_eq!(at.len(), gold.len());
            masked += at.len();
            total += ids.len() - 2;
        }
        let rate = masked as f64 / total as f64;
        assert!((0.08..0.25).contains(&rate), "masking rate {rate}");
    }
}

#[cfg(test)]
mod gradient_tests {
    use super::*;

    /// Every weight, in both layouts, must receive a gradient from the
    /// masked-word loss (a fused op without a backward pass once cut the
    /// encoder off silently, leaving only the output layer training).
    #[test]
    fn every_weight_gets_a_gradient() {
        for pre_ln in [false, true] {
            let sents: Vec<Vec<String>> = ["the cat sat on the mat .", "a dog ran in the park ."].iter().map(|l| l.split(' ').map(String::from).collect()).collect();
            let mut vocab: Vec<String> = crate::wordpiece::SPECIALS.iter().map(|s| s.to_string()).collect();
            for w in sents.iter().flatten() { if !vocab.contains(w) { vocab.push(w.clone()) } }
            let dir = std::env::temp_dir().join(format!("scribe-grad-{pre_ln}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("vocab.txt"), vocab.join("\n")).unwrap();
            std::fs::write(dir.join("config.json"), serde_json::json!({"hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
                "intermediate_size": 32, "max_position_embeddings": 16, "layer_norm_eps": 1e-12, "pre_layer_norm": pre_ln}).to_string()).unwrap();
            let (bert, vars) = Bert::new_random(&dir, Device::Cpu).unwrap();
            let rows: Vec<Vec<u32>> = sents.iter().map(|s| bert.encode_sentence(s)).collect();
            let at: Vec<(usize, usize)> = vec![(0, 2), (1, 3)];
            let gold: Vec<u32> = at.iter().map(|&(r, c)| rows[r][c]).collect();
            let mut masked = rows.clone();
            for &(r, c) in &at { masked[r][c] = bert.special("[MASK]") }
            let grads = masked_loss(&bert, &masked, &at, &gold, &Device::Cpu).unwrap().backward().unwrap();
            let missing: Vec<String> = vars.data().lock().unwrap().iter()
                .filter(|(n, v)| !n.contains("token_type") && grads.get(v.as_tensor()).is_none())
                .map(|(n, _)| n.clone()).collect();
            assert!(missing.is_empty(), "pre_ln={pre_ln}: no gradient for {missing:?}");
        }
    }
}
