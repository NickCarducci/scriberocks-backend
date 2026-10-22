//! Swift/Kotlin bindings via UniFFI. Thin wrappers over `Engine`; no logic here.

use crate::Engine;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, uniffi::Error)]
pub enum ScribeError {
    ModelsUnavailable { message: String },
}

impl std::fmt::Display for ScribeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScribeError::ModelsUnavailable { message } => write!(f, "couldn't load models: {message}"),
        }
    }
}

/// UTF-8 byte range into a variant's text.
#[derive(uniffi::Record)]
pub struct ByteRange {
    pub start: u32,
    pub end: u32,
}

#[derive(uniffi::Record)]
pub struct RockVariant {
    pub text: String,
    /// "trim", "cut", "reorder", "swap", "reframe"
    pub ops: Vec<String>,
    pub fluency_delta: f64,
    /// Words not in the original, as UTF-8 byte ranges into `text`.
    pub changed: Vec<ByteRange>,
}

#[derive(uniffi::Record)]
pub struct RockFrame {
    pub text: String,
    pub subject: Option<String>,
    pub modality: Option<String>,
    pub negated: bool,
    pub verb: Option<String>,
    pub object: Option<String>,
    pub link: Option<String>,
    pub question: bool,
    pub categories: Vec<String>,
    pub concepts: Vec<String>,
}

#[derive(uniffi::Object)]
pub struct ScribeEngine {
    engine: Engine,
    /// ScribeIE switch (off by default), set from the app's settings.
    scribe_ie: std::sync::atomic::AtomicBool,
}

fn name<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_value(v).ok().and_then(|j| j.as_str().map(String::from)).unwrap_or_default()
}

#[uniffi::export]
impl ScribeEngine {
    /// Loads every model from a directory (lexicon.bin, tagger.bin, …). Takes ~1 s; do it off the main thread.
    #[uniffi::constructor]
    pub fn new(models_dir: String) -> Result<Arc<Self>, ScribeError> {
        Engine::load(Path::new(&models_dir))
            .map(|engine| Arc::new(ScribeEngine { engine, scribe_ie: std::sync::atomic::AtomicBool::new(false) }))
            .map_err(|e| ScribeError::ModelsUnavailable { message: e.to_string() })
    }

    /// Turns ScribeIE (clarity re-ranking) on or off. Off by default.
    pub fn set_scribe_ie(&self, enabled: bool) {
        self.scribe_ie.store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether a ScribeIE clarity model is loaded (the setting does nothing without one).
    pub fn scribe_ie_available(&self) -> bool {
        self.engine.clarity.is_some()
    }

    pub fn variants(&self, rock: String, limit: u32) -> Vec<RockVariant> {
        let options = crate::scribe_ie::EditOptions { scribe_ie: self.scribe_ie.load(std::sync::atomic::Ordering::Relaxed), ..Default::default() };
        self.engine
            .variants_with(&rock, limit as usize, &options)
            .into_iter()
            .map(|v| RockVariant {
                text: v.text,
                ops: v.ops.iter().map(name).collect(),
                fluency_delta: v.fluency_delta,
                changed: v.changed.into_iter().map(|(a, b)| ByteRange { start: a as u32, end: b as u32 }).collect(),
            })
            .collect()
    }

    pub fn frames(&self, rock: String) -> Vec<RockFrame> {
        self.engine
            .frames(&rock)
            .into_iter()
            .map(|f| RockFrame {
                text: f.text,
                subject: f.subject,
                modality: f.modality.as_ref().map(name),
                negated: f.negated,
                verb: f.verb,
                object: f.object,
                link: f.link.as_ref().map(name),
                question: f.question,
                categories: f.categories,
                concepts: f.concepts,
            })
            .collect()
    }
}
