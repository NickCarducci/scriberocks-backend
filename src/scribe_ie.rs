//! ScribeIE: an optional fifth stage of ScribeSLM, off by default. After
//! proposing and judging, it re-ranks the judged suggestions by clarity: how
//! much clearer each version reads than the original. Its scorer learns from
//! editors' revisions (IteraTeR's clarity edits), a different kind of source
//! from ScribeSLM's standards, which is why it is opt-in. It never proposes;
//! it only reorders what ScribeSLM already judged.

/// Says how much clearer `after` is than `before` (any scale; 0 = no change).
/// None when it can't judge, which leaves the order alone.
pub trait ClarityScorer: Send + Sync {
    fn clarity(&self, before: &str, after: &str) -> Option<f64>;
}

/// Options for one request for suggestions. Default: ScribeIE off.
#[derive(Clone, Debug)]
pub struct EditOptions {
    /// Re-rank by clarity (needs a loaded scorer; otherwise no effect).
    pub scribe_ie: bool,
    /// Weight of clarity against fluency in the ranking.
    pub clarity_weight: f64,
}

impl Default for EditOptions {
    fn default() -> Self {
        EditOptions { scribe_ie: false, clarity_weight: 1.0 }
    }
}
