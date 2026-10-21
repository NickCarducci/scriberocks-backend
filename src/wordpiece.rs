//! Learns a WordPiece vocabulary (BERT format: a word's first piece as is,
//! later pieces prefixed "##") from our own corpus, by merging the most
//! frequent adjacent pair of pieces until the vocabulary is full (as BPE does;
//! tokenizing is then greedy longest-match, as `bert::Bert::word_pieces`).
//! Specials come first; [PAD] is id 0.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub const SPECIALS: [&str; 5] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

/// A vocabulary of at most `size` entries from lowercased word counts.
/// Words seen fewer than `min_count` times are left out of the merges (their
/// letters still count), so rare names don't take vocabulary slots.
pub fn train(counts: &HashMap<String, u64>, size: usize, min_count: u64) -> Vec<String> {
    let mut vocab: Vec<String> = SPECIALS.iter().map(|s| s.to_string()).collect();
    let mut id: HashMap<String, u32> = HashMap::new();
    let mut intern = |s: String, vocab: &mut Vec<String>| -> u32 {
        *id.entry(s.clone()).or_insert_with(|| { vocab.push(s); vocab.len() as u32 - 1 })
    };
    // Every character, as a first piece and as a continuation.
    let mut words: Vec<(Vec<u32>, u64)> = Vec::new();
    let mut letters: Vec<(char, u64)> = Vec::new();
    {
        let mut tally: HashMap<char, u64> = HashMap::new();
        for (w, &c) in counts {
            for ch in w.chars() { *tally.entry(ch).or_default() += c }
        }
        letters.extend(tally);
        letters.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    }
    for &(ch, _) in &letters {
        intern(ch.to_string(), &mut vocab);
        intern(format!("##{ch}"), &mut vocab);
        if vocab.len() >= size { vocab.truncate(size); return vocab }
    }
    let mut sorted: Vec<(&String, &u64)> = counts.iter().filter(|(_, c)| **c >= min_count).collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (w, &c) in sorted {
        let pieces: Vec<u32> = w.chars().enumerate().map(|(i, ch)| intern(if i == 0 { ch.to_string() } else { format!("##{ch}") }, &mut vocab)).collect();
        words.push((pieces, c));
    }
    // Pair counts, which words hold each pair, and a heap of (count, pair).
    let mut pairs: HashMap<(u32, u32), u64> = HashMap::new();
    let mut holders: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
    for (i, (p, c)) in words.iter().enumerate() {
        for w in p.windows(2) {
            *pairs.entry((w[0], w[1])).or_default() += c;
            holders.entry((w[0], w[1])).or_default().insert(i);
        }
    }
    let mut heap: BinaryHeap<(u64, Reverse<(u32, u32)>)> = pairs.iter().map(|(&p, &c)| (c, Reverse(p))).collect();
    while vocab.len() < size {
        let Some((count, Reverse(pair))) = heap.pop() else { break };
        if pairs.get(&pair).copied() != Some(count) || count == 0 { continue } // stale entry
        let joined = format!("{}{}", vocab[pair.0 as usize], vocab[pair.1 as usize].trim_start_matches("##"));
        let new = intern(joined, &mut vocab);
        for i in holders.remove(&pair).unwrap_or_default() {
            let (old, c) = (words[i].0.clone(), words[i].1);
            let mut merged = Vec::with_capacity(old.len());
            let mut k = 0;
            while k < old.len() {
                if k + 1 < old.len() && (old[k], old[k + 1]) == pair { merged.push(new); k += 2 } else { merged.push(old[k]); k += 1 }
            }
            if merged.len() == old.len() { continue }
            for w in old.windows(2) {
                if let Some(x) = pairs.get_mut(&(w[0], w[1])) { *x = x.saturating_sub(c) }
            }
            for w in merged.windows(2) {
                let p = (w[0], w[1]);
                let x = pairs.entry(p).or_default();
                *x += c;
                heap.push((*x, Reverse(p)));
                holders.entry(p).or_default().insert(i);
            }
            words[i].0 = merged;
        }
        pairs.remove(&pair);
    }
    vocab
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(text: &str) -> HashMap<String, u64> {
        let mut c = HashMap::new();
        for w in text.split_whitespace() { *c.entry(w.to_string()).or_default() += 1 }
        c
    }

    #[test]
    fn specials_first_with_pad_at_zero_and_size_respected() {
        let v = train(&counts("low lower lowest newer newest wider widest low low"), 40, 1);
        assert_eq!(&v[..5], &SPECIALS.map(String::from));
        assert!(v.len() <= 40);
        assert_eq!(v.iter().collect::<HashSet<_>>().len(), v.len(), "no duplicates");
    }

    #[test]
    fn merges_frequent_words_and_endings() {
        let text = "low low low low lower lowest newest newest widest widest walked talked walked";
        let v = train(&counts(text), 200, 1);
        assert!(v.contains(&"low".to_string()), "{v:?}");
        assert!(v.iter().any(|p| p == "##est" || p.ends_with("est")), "{v:?}");
        // Every continuation piece is marked; first pieces aren't.
        assert!(v[5..].iter().all(|p| !p.starts_with("##") || p.len() > 2));
    }

    #[test]
    fn rare_words_do_not_take_merges() {
        let v = train(&counts("zyxw cat cat cat"), 100, 2);
        assert!(v.contains(&"cat".to_string()));
        assert!(!v.contains(&"zyxw".to_string()));
        assert!(v.contains(&"z".to_string()), "its letters still count");
    }
}
