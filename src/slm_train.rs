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

/// Consecutive groups of rows (in order) each holding at most `max_pieces`
/// padded pieces (rows × longest row in the group); one row at least. With
/// `max_pieces` = 0, one group of everything.
pub fn micro_batches(lengths: &[usize], max_pieces: usize) -> Vec<Vec<usize>> {
    if max_pieces == 0 { return vec![(0..lengths.len()).collect()] }
    let (mut out, mut cur, mut longest) = (Vec::new(), Vec::<usize>::new(), 0);
    for (i, &l) in lengths.iter().enumerate() {
        let m = longest.max(l);
        if !cur.is_empty() && (cur.len() + 1) * m > max_pieces {
            out.push(std::mem::take(&mut cur));
            longest = 0;
        }
        longest = longest.max(l);
        cur.push(i);
    }
    if !cur.is_empty() { out.push(cur) }
    out
}

/// One training step's gradients by gradient accumulation: the masked-word
/// loss over `rows` (masks already applied, `at`/`gold` over the whole batch)
/// plus `joint` × the dictionary loss over `dict`, run in micro-batches of at
/// most `max_pieces` padded pieces, each normalized by the WHOLE batch's count
/// (hidden words; dictionary examples), so the summed gradients equal the full
/// batch's (only rounding differs). Bounds a step's working set: a 128 × 128-
/// piece batch's intermediates (2,400 hidden words × 30,000 vocabulary) held
/// ~9 GB of GPU memory. Returns the gradients and the batch loss (detached).
#[allow(clippy::too_many_arguments)]
pub fn accumulated_grads(bert: &Bert, vars: &[candle_core::Var], rows: &[Vec<u32>], at: &[(usize, usize)], gold: &[u32],
                         dict: &[&DictExample], joint: f64, max_pieces: usize, dev: &Device) -> Result<(candle_core::backprop::GradStore, Tensor)> {
    let mut acc: Option<candle_core::backprop::GradStore> = None;
    let mut total = Tensor::zeros((), candle_core::DType::F32, dev)?;
    let mut add = |loss: Tensor, acc: &mut Option<candle_core::backprop::GradStore>| -> Result<()> {
        let g = loss.backward()?;
        total = (&total + loss.detach())?;
        match acc {
            None => *acc = Some(g),
            Some(a) => for v in vars {
                if let Some(x) = g.get(v.as_tensor()) {
                    let sum = match a.get(v.as_tensor()) { Some(y) => (y + x)?, None => x.clone() };
                    a.insert(v.as_tensor(), sum);
                }
            },
        }
        Ok(())
    };
    let n = at.len().max(1) as f64;
    for group in micro_batches(&rows.iter().map(Vec::len).collect::<Vec<_>>(), max_pieces) {
        let index: std::collections::HashMap<usize, usize> = group.iter().enumerate().map(|(k, &r)| (r, k)).collect();
        let (sub_at, sub_gold): (Vec<(usize, usize)>, Vec<u32>) = at.iter().zip(gold).filter_map(|(&(r, c), &g)| index.get(&r).map(|&k| ((k, c), g))).unzip();
        if sub_at.is_empty() { continue }
        let sub_rows: Vec<Vec<u32>> = group.iter().map(|&r| rows[r].clone()).collect();
        add((masked_loss_sum(bert, &sub_rows, &sub_at, &sub_gold, dev)? / n)?, &mut acc)?;
    }
    if joint > 0.0 && !dict.is_empty() {
        let m = dict.len() as f64;
        for group in micro_batches(&dict.iter().map(|e| e.ids.len()).collect::<Vec<_>>(), max_pieces / 2) {
            let sub: Vec<&DictExample> = group.iter().map(|&i| dict[i]).collect();
            add((dictionary_loss_sum(bert, &sub, dev)? * (joint / m))?, &mut acc)?;
        }
    }
    let grads = acc.unwrap_or_else(|| total.backward().expect("an empty step"));
    Ok((grads, total))
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

/// Summed negative log-probability of the original pieces at the masked positions.
pub fn masked_loss_sum(bert: &Bert, rows: &[Vec<u32>], at: &[(usize, usize)], gold: &[u32], dev: &Device) -> Result<Tensor> {
    let lp = bert.masked_log_probs(rows, at)?;
    lp.gather(&Tensor::new(gold, dev)?.unsqueeze(1)?, 1)?.sum_all()?.neg()
}

/// Mean negative log-probability of the original pieces at the masked positions.
pub fn masked_loss(bert: &Bert, rows: &[Vec<u32>], at: &[(usize, usize)], gold: &[u32], dev: &Device) -> Result<Tensor> {
    masked_loss_sum(bert, rows, at, gold, dev)? / at.len().max(1) as f64
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

/// Mean −log of the probability mass each hidden slot puts on its acceptable words.
pub fn dictionary_loss(bert: &Bert, batch: &[&DictExample], dev: &Device) -> Result<Tensor> {
    dictionary_loss_sum(bert, batch, dev)? / batch.len().max(1) as f64
}

/// Summed −log of the probability mass each hidden slot puts on its acceptable words.
pub fn dictionary_loss_sum(bert: &Bert, batch: &[&DictExample], dev: &Device) -> Result<Tensor> {
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
    lse.neg()?.sum_all()
}

#[cfg(test)]
mod accumulation_tests {
    use super::*;

    /// Gradient accumulation must reproduce the full batch's gradients.
    #[test]
    fn accumulated_gradients_equal_the_full_batch() {
        let lines = ["the cat sat on the mat .", "a dog ran in the big green park today .", "she drank tea .",
                     "we built a small boat on the river with our old friends last summer .", "birds sang ."];
        let sents: Vec<Vec<String>> = lines.iter().map(|l| l.split(' ').map(String::from).collect()).collect();
        let mut vocab: Vec<String> = crate::wordpiece::SPECIALS.iter().map(|s| s.to_string()).collect();
        for w in sents.iter().flatten() { if !vocab.contains(w) { vocab.push(w.clone()) } }
        let dir = std::env::temp_dir().join("scribe-accumulate");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("vocab.txt"), vocab.join("\n")).unwrap();
        std::fs::write(dir.join("config.json"), serde_json::json!({"hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
            "intermediate_size": 32, "max_position_embeddings": 32, "layer_norm_eps": 1e-12, "pre_layer_norm": true}).to_string()).unwrap();
        let (bert, vars) = Bert::new_random(&dir, Device::Cpu).unwrap();
        let vars = vars.all_vars();
        let data: Vec<Encoded> = sents.iter().map(|s| { let (ids, first) = bert.encode_words(s); Encoded { ids, first: first.iter().map(|&f| f as u16).collect() } }).collect();
        let refs: Vec<&Encoded> = data.iter().collect();
        let mut rng = Rng(11);
        let (rows, at, gold) = loop { let m = mask_whole_words(&bert, &refs, &mut rng); if m.1.len() >= 4 { break m } };
        let dict: Vec<DictExample> = data.iter().map(|e| DictExample { ids: e.ids.clone(), at: 2, accept: vec![e.ids[2], e.ids[1]] }).collect();
        let dref: Vec<&DictExample> = dict.iter().collect();
        let dev = Device::Cpu;
        let full = (masked_loss(&bert, &rows, &at, &gold, &dev).unwrap() + (dictionary_loss(&bert, &dref, &dev).unwrap() * 0.3).unwrap()).unwrap();
        let g_full = full.backward().unwrap();
        let (g_acc, total) = accumulated_grads(&bert, &vars, &rows, &at, &gold, &dref, 0.3, 30, &dev).unwrap();
        assert!(micro_batches(&rows.iter().map(Vec::len).collect::<Vec<_>>(), 30).len() > 1, "really split");
        assert!((total.to_scalar::<f32>().unwrap() - full.to_scalar::<f32>().unwrap()).abs() < 1e-5);
        for v in &vars {
            let (Some(a), Some(b)) = (g_full.get(v.as_tensor()), g_acc.get(v.as_tensor())) else { continue };
            let diff = (a - b).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            let scale = a.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap().max(1e-6);
            assert!(diff <= 1e-5 + 1e-4 * scale, "gradient differs by {diff} (scale {scale})");
        }
    }

    #[test]
    fn micro_batches_keep_order_and_respect_the_budget() {
        let lens = [10, 10, 40, 10, 100, 5];
        let g = micro_batches(&lens, 60);
        assert_eq!(g.concat(), vec![0, 1, 2, 3, 4, 5]);
        for b in &g { assert!(b.len() == 1 || b.len() * b.iter().map(|&i| lens[i]).max().unwrap() <= 60, "{g:?}") }
        assert_eq!(micro_batches(&lens, 0), vec![vec![0, 1, 2, 3, 4, 5]]);
    }
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
