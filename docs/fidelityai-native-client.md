# FidelityAI prose models in a Swift or Rust app (notes, October 24, 2026)

**When:** only once FidelityAI earns a place in Scriberocks. North star: held-out
Good/Very good paragraphs against ordinary ones (`scripts/fidelity_certified_eval.py`),
document-level AUC ≥ 0.70 with length controlled. Today: 0.564.

**Why it's cheap:** role embeddings are learned on the backend and shipped in the
artifact as JSON (`grammar.encoding.tables`: per role, token id → 16 numbers, plus
`unknown`). A client only looks vectors up. The work a client must reproduce
exactly is preprocessing: text → frames → role values.

**Done (core `src/fidelity_frames.rs`):** path B's frame rules ("scribe-v1"), over
scribe's own tagger and parser pruned at 1.0 (`prepare`), identical to FidelityAI's
`scribe_parse.js`. Kept identical by `tests/fixtures/fidelity_frames.jsonl`
(regenerate with FidelityAI `engine/scripts/frames_fixture.mjs`).

**Left for a native client:**
1. **Frames → token ids:** for each frame, each of the artifact's axes in order
   (`grammar.roles`, prefix before `=`): `axis=value` if present in `roles`, else
   `axis=other`, or `axis=missing` when the slot is empty (FidelityAI textEncode.js
   `encodeAgainstRoles`). Only for artifacts with `frameExtractor: "scribe-v1"`.
2. **Scorer:** port `scorer.js` for role-embedding artifacts: tuples of k tokens
   (within `windowArity` when set), each tuple's tensor product of role vectors
   (`unknown` for unseen ids), projection by `clientHandoff.handoff.T`, then the
   primary score (`ocsvm_fit_distance`) from support vectors and dual coefficients.
   Check against `scoreEncodedSequences` on the demo artifacts.
3. **Size:** the Rust tagger, parser and dictionary are already in the app; nothing new to ship.
