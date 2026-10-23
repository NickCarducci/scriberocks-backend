//! ScribeIE (Intent-conditioned Editing): ScribeSLM's optional generative
//! stage, behind the app's "Generate" toggle (off by default). When on, every
//! proposal ScribeSLM made is enhanced: ScribeIE rewrites it toward the
//! writer's intent (clarity first), and the writer may save either version.
//! ScribeSLM's proposals, text and order are never changed; the enhanced
//! version rides alongside each one (`Variant::enhanced`), with the words
//! ScribeIE added marked (`Variant::enhanced_changed`). Its model will learn
//! from editors' revisions (IteraTeR) — a different source from ScribeSLM's
//! standards, which is why it is opt-in.

/// ScribeIE's rewrite of one proposal of `rock`; None leaves it as is.
pub trait Enhancer: Send + Sync {
    fn enhance(&self, rock: &str, proposal: &str) -> Option<String>;
}

/// Options for one request for suggestions. Default: Generate off, no style.
#[derive(Clone, Debug, Default)]
pub struct EditOptions {
    /// Enhance every proposal with ScribeIE (needs a loaded enhancer; otherwise no effect).
    pub generate: bool,
    /// The writer's prose style: suggestions are judged by that style's model
    /// (if loaded); otherwise by the default judge. Another style's judge is never used.
    pub style: Option<crate::engine::ProseStyle>,
    /// The writer's variety slider, 0 (fewer, safer swaps) to 1 (more, bolder):
    /// maps onto the swap cut-off (`Engine::margin_for`). None = the engine's default.
    pub variety: Option<f32>,
}

/// Byte ranges of the words in `enhanced` that the plain `proposal` doesn't
/// have (case-insensitive, counting repeats), for marking ScribeIE's part.
pub fn added_words(proposal: &str, enhanced: &str) -> Vec<(usize, usize)> {
    let norm = |w: &str| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'').to_lowercase();
    let mut have: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for w in proposal.split_whitespace() { *have.entry(norm(w)).or_default() += 1 }
    let mut out = Vec::new();
    let mut at = 0;
    for w in enhanced.split_whitespace() {
        let start = at + enhanced[at..].find(w).unwrap_or(0);
        at = start + w.len();
        let key = norm(w);
        match have.get_mut(&key) {
            Some(n) if *n > 0 => *n -= 1,
            _ if !key.is_empty() => {
                let lead = w.len() - w.trim_start_matches(|c: char| !c.is_alphanumeric()).len();
                let core = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
                out.push((start + lead, start + lead + core.len()));
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn added_words_marks_only_new_words_with_their_punctuation_trimmed() {
        let e = "I can't come to the celebration, sadly.";
        let r = added_words("I can't come to the party.", e);
        let got: Vec<&str> = r.iter().map(|&(a, b)| &e[a..b]).collect();
        assert_eq!(got, vec!["celebration", "sadly"]);
        assert!(added_words("same words here", "Same words here").is_empty());
    }
}
