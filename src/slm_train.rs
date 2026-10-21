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

/// Mean negative log-probability of the original pieces at the masked positions.
pub fn masked_loss(bert: &Bert, rows: &[Vec<u32>], at: &[(usize, usize)], gold: &[u32], dev: &Device) -> Result<Tensor> {
    let lp = bert.masked_log_probs(rows, at)?;
    lp.gather(&Tensor::new(gold, dev)?.unsqueeze(1)?, 1)?.neg()?.mean_all()
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
mod tests {
    use super::*;

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
