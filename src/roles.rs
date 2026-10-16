//! Scriberocks' preprocessing for FidelityAI: a variant becomes an integer
//! role sequence (the "sequence_tokens" source FidelityAI accepts). Only
//! these integers ever leave the device — never the rock's text.
//!
//! The vocabulary is fixed and append-only: ids must not change between
//! app versions, or fitted artifacts would score the wrong roles.

use crate::variants::Op;

/// Append-only. Index = role id.
pub const ROLES: &[&str] = &[
    // 0–6: operations
    "op:trim", "op:cut", "op:reorder", "op:swap", "op:reframe", "op:connect", "op:punctuate",
    // 7–9: length change vs the original
    "len:shorter", "len:same", "len:longer",
    // 10–15: word class of each changed word
    "chg:NOUN", "chg:VERB", "chg:ADJ", "chg:ADV", "chg:PUNCT", "chg:OTHER",
];

pub fn role_id(name: &str) -> Option<u32> {
    ROLES.iter().position(|r| *r == name).map(|i| i as u32)
}

fn op_role(op: Op) -> &'static str {
    match op {
        Op::Trim => "op:trim",
        Op::Cut => "op:cut",
        Op::Reorder => "op:reorder",
        Op::Swap => "op:swap",
        Op::Reframe => "op:reframe",
        Op::Connect => "op:connect",
        Op::Punctuate => "op:punctuate",
    }
}

/// Ops in order, then the length bucket, then each changed word's class (UPOS).
pub fn encode(ops: &[Op], original_words: usize, variant_words: usize, changed_upos: &[&str]) -> Vec<u32> {
    let mut seq: Vec<u32> = ops.iter().map(|&op| role_id(op_role(op)).unwrap()).collect();
    let len = match variant_words.cmp(&original_words) {
        std::cmp::Ordering::Less => "len:shorter",
        std::cmp::Ordering::Equal => "len:same",
        std::cmp::Ordering::Greater => "len:longer",
    };
    seq.push(role_id(len).unwrap());
    for upos in changed_upos {
        let name = match *upos {
            "NOUN" | "PROPN" => "chg:NOUN",
            "VERB" | "AUX" => "chg:VERB",
            "ADJ" => "chg:ADJ",
            "ADV" => "chg:ADV",
            "PUNCT" => "chg:PUNCT",
            _ => "chg:OTHER",
        };
        seq.push(role_id(name).unwrap());
    }
    seq
}

/// One FidelityAI source record ("sequence_tokens" path).
pub fn source_record(sequence: &[u32]) -> String {
    serde_json::json!({ "sequence": sequence }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variants::Op;

    #[test]
    fn vocabulary_is_stable_and_append_only() {
        // These ids are a contract with fitted artifacts. Never reorder.
        assert_eq!(ROLES[0], "op:trim");
        assert_eq!(role_id("op:punctuate"), Some(6));
        assert_eq!(role_id("len:shorter"), Some(7));
        assert_eq!(role_id("chg:NOUN"), Some(10));
        assert!(ROLES.len() < 64);
    }

    #[test]
    fn encodes_ops_then_length_then_changed_word_classes() {
        let seq = encode(&[Op::Trim, Op::Reframe], 12, 10, &["VERB", "PART"]);
        assert_eq!(
            seq.iter().map(|&i| ROLES[i as usize]).collect::<Vec<_>>(),
            vec!["op:trim", "op:reframe", "len:shorter", "chg:VERB", "chg:OTHER"]
        );
    }

    #[test]
    fn same_variant_same_sequence_and_length_buckets() {
        assert_eq!(encode(&[Op::Swap], 8, 8, &["ADJ"]), encode(&[Op::Swap], 8, 8, &["ADJ"]));
        assert_eq!(ROLES[encode(&[Op::Connect], 8, 9, &[])[1] as usize], "len:longer");
        assert_eq!(ROLES[encode(&[Op::Cut], 8, 8, &[])[1] as usize], "len:same");
    }

    #[test]
    fn source_records_are_fidelityai_sequence_json() {
        let line = source_record(&[0, 7, 10]);
        assert_eq!(line, r#"{"sequence":[0,7,10]}"#);
    }
}
