//! Scriberocks language core — a dictionary and a grammar, no pretrained language model.
//!
//! Layer 1 — `lexicon`: Open English WordNet (senses, definitions, 45 categories).
//! Layer 2 — `tagger`, `parser`: grammar learned from the UD English EWT treebank.
//! Layer 3 — `wsd`: choose a word's sense by overlap with dictionary definitions.
//! On top — `frames` (what a rock says) and `variants` (other cuts of a rock),
//! with `inflect` and a dictionary-trained fluency model (`lm`).

pub mod bert;
pub mod conllu;
pub mod engine;
pub mod ffi;
pub mod frames;
pub mod inflect;
pub mod kgram;
pub mod lexicon;
pub mod lexsub;
pub mod lm;
pub mod parser;
pub mod punctuate;
pub mod roles;
pub mod semcor;
pub mod syntax;
pub mod tagger;
pub mod text;
pub mod usage;
pub mod variants;
pub mod wsd;

pub use engine::Engine;
pub use frames::{Frame, Link, Modality};
pub use lexicon::{Lexicon, Pos};
pub use variants::{Op, Variant};

uniffi::setup_scaffolding!();
