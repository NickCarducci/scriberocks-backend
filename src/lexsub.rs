//! Lexical-substitution metrics, as the SWORDS benchmark (Stanford Word
//! Substitution Benchmark, Lee et al. 2021) computes them in its
//! swords/eval.py: generalized average precision (GAP) for ranking a given
//! candidate list, and F@k for proposing substitutes.
use std::collections::HashSet;

/// GAP for one target. `gold`: every rated candidate with its weight (share
/// of judges who accepted it); `ranked`: the system's candidates, best
/// first. Candidates the gold list doesn't rate weigh 0 but still take a rank.
pub fn gap(gold: &[(String, f64)], ranked: &[String]) -> f64 {
    let weight = |s: &str| gold.iter().find(|(g, _)| g == s).map_or(0.0, |(_, w)| *w);
    let average_precision = |weights: &[f64]| {
        let (mut sum, mut cumulative) = (0.0, 0.0);
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if w > 0.0 {
                sum += cumulative / (i + 1) as f64;
            }
        }
        sum
    };
    let mut ideal: Vec<f64> = gold.iter().map(|(_, w)| *w).collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let best = average_precision(&ideal);
    if best == 0.0 {
        return 0.0;
    }
    average_precision(&ranked.iter().map(|s| weight(s)).collect::<Vec<_>>()) / best
}

/// Precision, recall and F at `k`, pooled over targets as swords' pr_at_k
/// does: hits / outputs, hits / Σ min(|accepted|, k). Each element pairs a
/// target's accepted substitutes with the system's list, best first.
pub fn pr_at_k(targets: &[(HashSet<String>, Vec<String>)], k: usize) -> (f64, f64, f64) {
    let (mut hits, mut outputs, mut possible) = (0usize, 0usize, 0usize);
    for (accepted, ranked) in targets {
        let top = &ranked[..ranked.len().min(k)];
        hits += top.iter().filter(|s| accepted.contains(*s)).count();
        outputs += top.len();
        possible += accepted.len().min(k);
    }
    let p = if outputs == 0 { 0.0 } else { hits as f64 / outputs as f64 };
    let r = if possible == 0 { 0.0 } else { hits as f64 / possible as f64 };
    let f = if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) };
    (100.0 * p, 100.0 * r, 100.0 * f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gold(items: &[(&str, f64)]) -> Vec<(String, f64)> {
        items.iter().map(|(s, w)| (s.to_string(), *w)).collect()
    }
    fn list(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn gap_is_one_for_the_ideal_order_and_lower_otherwise() {
        let g = gold(&[("a", 1.0), ("b", 0.5), ("c", 0.0)]);
        assert!((gap(&g, &list(&["a", "b", "c"])) - 1.0).abs() < 1e-9);
        // Reversed: weights 0, .5, 1 → (0.5/2 + 1.5/3) / (1/1 + 1.5/2) = 0.75 / 1.75
        assert!((gap(&g, &list(&["c", "b", "a"])) - 0.75 / 1.75).abs() < 1e-9);
    }

    #[test]
    fn unrated_candidates_take_a_rank_but_add_nothing() {
        let g = gold(&[("a", 1.0)]);
        assert!((gap(&g, &list(&["x", "a"])) - 0.5).abs() < 1e-9);
        assert_eq!(gap(&gold(&[("a", 0.0)]), &list(&["a"])), 0.0);
    }

    #[test]
    fn f_at_k_pools_hits_over_targets_and_caps_recall_at_k() {
        let accepted: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let targets = vec![(accepted.clone(), list(&["a", "x"])), (accepted, list(&["b", "c", "a"]))];
        let (p, r, f) = pr_at_k(&targets, 2);
        // top-2 lists: [a, x] → 1 hit; [b, c] → 2 hits. P = 3/4, R = 3/(2+2).
        assert!((p - 75.0).abs() < 1e-9 && (r - 75.0).abs() < 1e-9 && (f - 75.0).abs() < 1e-9);
    }
}
