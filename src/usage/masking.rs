//! Masked-word training examples (BERT-style 80/10/10).

use crate::usage::vocab::{MASK, SPECIALS};

/// xorshift64*: small, fast, reproducible.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

pub struct Masked {
    pub input: Vec<u32>,
    /// The original id at masked positions; None elsewhere.
    pub labels: Vec<Option<u32>>,
}

/// Picks ~15% of positions; of those 80% become [MASK], 10% a random word,
/// 10% stay. At least one position is always picked.
pub fn mask(ids: &[u32], vocab_size: u32, rng: &mut Rng) -> Masked {
    let mut input = ids.to_vec();
    let mut labels = vec![None; ids.len()];
    let pick = |i: usize, input: &mut Vec<u32>, labels: &mut Vec<Option<u32>>, rng: &mut Rng| {
        labels[i] = Some(ids[i]);
        let r = rng.unit();
        if r < 0.8 {
            input[i] = MASK;
        } else if r < 0.9 {
            input[i] = SPECIALS.len() as u32 + rng.below((vocab_size - SPECIALS.len() as u32) as u64) as u32;
        }
    };
    for i in 0..ids.len() {
        if rng.unit() < 0.15 {
            pick(i, &mut input, &mut labels, rng);
        }
    }
    if labels.iter().all(Option::is_none) && !ids.is_empty() {
        let i = rng.below(ids.len() as u64) as usize;
        pick(i, &mut input, &mut labels, rng);
    }
    Masked { input, labels }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usage::vocab::{MASK, PAD};

    #[test]
    fn masks_about_fifteen_percent_and_labels_only_masked_positions() {
        let ids: Vec<u32> = (10..1010).collect();
        let mut rng = Rng::new(7);
        let ex = mask(&ids, 2000, &mut rng);
        let masked = ex.labels.iter().filter(|l| l.is_some()).count();
        assert!((120..=180).contains(&masked), "masked {masked} of 1000");
        for (i, label) in ex.labels.iter().enumerate() {
            match label {
                Some(original) => assert_eq!(*original, ids[i]),
                None => assert_eq!(ex.input[i], ids[i]),
            }
        }
        let replaced_with_mask = ex.labels.iter().zip(&ex.input).filter(|(l, x)| l.is_some() && **x == MASK).count();
        assert!(replaced_with_mask as f64 / masked as f64 > 0.65);
    }

    #[test]
    fn always_masks_at_least_one_and_never_pads() {
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            let ex = mask(&[11, 12], 100, &mut rng);
            assert!(ex.labels.iter().any(Option::is_some));
            assert!(!ex.input.contains(&PAD));
        }
    }

    #[test]
    fn rng_is_deterministic() {
        let (mut a, mut b) = (Rng::new(3), Rng::new(3));
        assert_eq!((0..5).map(|_| a.next()).collect::<Vec<_>>(), (0..5).map(|_| b.next()).collect::<Vec<_>>());
    }
}
