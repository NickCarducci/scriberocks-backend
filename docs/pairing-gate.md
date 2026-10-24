# Pairing gate: results (October 23, 2026)

**Idea.** Hold back a word's swaps when the meaning the engine picked for it
doesn't fit the words it's attached to (compositional anomaly detection).
Example: "fix run-on sentences" got *fix* in its "set, decide" meaning, so the
engine offered *define / set / limit*.

**Method** (`src/compose.rs`, `examples/compose_build.rs`). We parsed SemCor and
the Princeton WordNet Gloss Corpus with the engine's parser. Every link between
two sense-tagged words became a record (meaning, relation, partner meaning):
277k links, 148,866 (meaning, relation) entries, 2 MB. A partner counts as
typical when its hypernym ancestry overlaps (Jaccard) a partner seen in training.
- Absolute gate: hold back the swap when typicality falls below a threshold.
- Relative gate: hold back the swap when another meaning of the word fits the partner better by a margin.

**Test.** The human-judged acceptance rocks, each style with its own judge, top
20 suggestions per rock, default cut-off (2.5). We counted wanted substitutes
offered against forbidden ones that appeared.

| Setting | Casual: wanted / forbidden | Publishable: wanted / forbidden |
|---|---|---|
| No gate | 61/210 · 17/129 | 40/151 · 10/409 |
| Typical < 0.15 | 50 · 13 | 29 · 10 |
| Typical < 0.25 | 42 · 11 | 22 · 9 |
| Typical < 0.35 | 30 · 9 | 17 · 7 |
| Typical < 0.5 | 26 · 9 | 16 · 6 |
| Outcompeted by > 0.1 | 55 · 16 | 33 · 10 |
| Outcompeted by > 0.2 | 57 · 16 | 35 · 10 |
| Typical < 0.25, at least 10 partners seen | 54 · 16 | 28 · 9 |

**Result.** Not adopted (the gate stays off; `Engine::compose` is None).
- On casual rocks, the gate removes wanted and forbidden suggestions in about
  equal proportion, which is no better than removing swaps at random.
- On publishable rocks it is worse than random: wanted suggestions halve while
  forbidden ones barely drop.
- The existing cut-off does better. Lowering casual's margin from 2.5 to 1.0
  cut bad swaps by 41% for 28% of good ones (Oct 22 rocks run).
- On the user's paragraph, typical < 0.25 removed *define / set / limit*, but
  also *suggest*, *try* ("try this app out") and *learn*.

**Why.** The records are sparse: 277k links spread over 148,866 entries is
about 2 partners per (meaning, relation). Partner meanings at run time come from
the engine's own sense choices. Many forbidden rock substitutes are the right
meaning but a poor fit, which a meaning-pairing check can't see.

**FidelityAI.** We did not pay for a run. The same records encoded as
`role_embedding_v1` tuples (meaning, relation one-hot, partner category) would
change the kernel, not the sparsity. The October 18 postmortem found the
representation, not the kernel, to be the limit.
