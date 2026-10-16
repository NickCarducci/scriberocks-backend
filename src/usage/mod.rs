//! The usage model: a small transformer trained from scratch to fill in
//! hidden words, on usage sentences only (dictionary examples, Wiktionary
//! examples, treebank sentences) — no definitions, no web text.
//!
//! Used to rank dictionary-proposed swaps by how well they fit *this*
//! sentence, and to pick senses by comparing a word's context with each
//! sense's example sentences.

pub mod vocab;
pub mod masking;
pub mod model;
pub mod corpus;
pub mod senses;
