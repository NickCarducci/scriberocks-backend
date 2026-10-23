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
    /// ScribeIE's enhanced version (Generate on and a model loaded), else nil.
    pub enhanced: Option<String>,
    /// Words ScribeIE added, as UTF-8 byte ranges into `enhanced`.
    pub enhanced_changed: Vec<ByteRange>,
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
    /// The Generate toggle (ScribeIE; off by default), set from the app.
    generate: std::sync::atomic::AtomicBool,
    /// The writer's prose style (casual / publishable), set from the app.
    style: std::sync::Mutex<Option<crate::engine::ProseStyle>>,
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
            .map(|engine| Arc::new(ScribeEngine { engine, generate: std::sync::atomic::AtomicBool::new(false), style: std::sync::Mutex::new(None) }))
            .map_err(|e| ScribeError::ModelsUnavailable { message: e.to_string() })
    }

    /// The Generate toggle: ScribeIE enhances every proposal. Off by default.
    pub fn set_generate(&self, enabled: bool) {
        self.generate.store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether a ScribeIE model is loaded (the toggle does nothing without one).
    pub fn generate_available(&self) -> bool {
        self.engine.enhancer.is_some()
    }

    /// Sets the writer's prose style ("casual" or "publishable"); "" clears it.
    /// Returns false (and keeps the default judge) when that style's model
    /// isn't loaded, so the app only offers styles in `available_styles`.
    pub fn set_style(&self, name: String) -> bool {
        let style = crate::engine::ProseStyle::from_name(&name).filter(|s| self.engine.available_styles().contains(s));
        *self.style.lock().unwrap() = style;
        style.is_some() || name.is_empty()
    }

    /// Prose styles whose judge models are loaded ("casual", "publishable").
    pub fn available_styles(&self) -> Vec<String> {
        self.engine.available_styles().into_iter().map(|s| s.name().to_string()).collect()
    }

    pub fn variants(&self, rock: String, limit: u32) -> Vec<RockVariant> {
        let options = crate::scribe_ie::EditOptions {
            generate: self.generate.load(std::sync::atomic::Ordering::Relaxed),
            style: *self.style.lock().unwrap(),
        };
        self.engine
            .variants_with(&rock, limit as usize, &options)
            .into_iter()
            .map(|v| RockVariant {
                text: v.text,
                ops: v.ops.iter().map(name).collect(),
                fluency_delta: v.fluency_delta,
                changed: v.changed.into_iter().map(|(a, b)| ByteRange { start: a as u32, end: b as u32 }).collect(),
                enhanced: v.enhanced,
                enhanced_changed: v.enhanced_changed.into_iter().map(|(a, b)| ByteRange { start: a as u32, end: b as u32 }).collect(),
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
