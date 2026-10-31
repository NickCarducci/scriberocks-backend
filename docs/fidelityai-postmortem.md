# Postmortem: FidelityAI in Scriberocks

Two rounds: the sense picker (October 14–18, 2026) and the wider writing
pipeline (October 24–25, 2026, [second round](#second-round-can-fidelityai-help-the-writing-pipeline-at-all)).

In the first round, Scriberocks used FidelityAI's public API the way any
customer would, with no internal shortcuts. The goal was to see whether
FidelityAI could choose a word's meaning (its *sense*) for the editor.

## Summary

- **Outcome:** FidelityAI can't serve as Scriberocks' sense picker. On the
  real training set it was right 25–43.8% of the time. Always choosing the
  meaning WordNet lists first was right 65–68% of the time.
- **Cause:** the method, not FidelityAI. Every alternative tried, classical
  ones included, also scored at or below first-listed. Choosing a meaning by
  similarity to past examples is weak at this data size.
- **Where FidelityAI fits:** the editor's last step. That step asks whether a
  synonym option reads naturally in this sentence. A one-class model per
  option is the job FidelityAI is built for.
- **What FidelityAI gained:** the effort fixed billing, job-status and
  data-retention bugs that affected every customer. It also added a new
  encoding and offline scorers that any customer can use.
- **Second round:** on grammar, word fit, edit ranking, rock quality and style,
  FidelityAI didn't beat the judge or plain counts by a reliable margin. Its best
  grammar result was 0.54 against the judge's 0.74 (0.5 is chance). The round
  also found and fixed a score that ranked unfamiliar records as typical.

## Terms

- **Sense:** one dictionary meaning of a word, for example *bank* (river) versus *bank* (money).
- **WordNet:** the dictionary Scriberocks uses. It lists each word's meanings roughly from most to least common.
- **First-listed:** always pick a word's first WordNet meaning. This is the standard baseline.
- **SemCor:** a corpus of sentences in which every word is hand-tagged with its WordNet meaning. We train on its even-numbered documents and test on its odd-numbered ones.
- **ALL:** the standard combined test set from the Word Sense Disambiguation Evaluation Framework (Raganato and others, 2017).
- **Option-bearing sense:** a meaning that has at least one usable synonym. These are the only meanings the editor can suggest from.
- **Usage model:** Scriberocks' own small language model, trained on how words are used.
- **Sense head:** a small classifier on top of the usage model, trained on SemCor to pick meanings.
- **Reference bank:** the training samples FidelityAI compares every new input against.

## Timeline

| Date (2026) | Event |
|---|---|
| Oct 14–15 | Scriberocks account set up on FidelityAI. Pricing changed to a $5 floor per training run. API-key access moved to `/api/v1`, outside Cloudflare Access. |
| Oct 15–16 | First uploads failed: JSON Lines sources were rejected, the vocabulary didn't fit, and checkout broke on Stripe's metadata limits. Default capacity raised to 26 qubits. Runs that can't fit are now refused before payment. |
| Oct 16 | Fits at 26 qubits took about 3 hours, and jobs killed by the platform stayed "running" forever. Killed jobs are now reported as failed and retried for free. In-job time budgets were removed. |
| Oct 17 | Scoring moved fully to the customer's device: no server-side scoring after a fit, and inputs deleted after training by default. The portable scorer and text SDK shipped, and the public docs were rewritten. |
| Oct 17 | Smoke model (2 nouns, 4 senses) scored every sense identically. Cause: FidelityAI's default features pool nearby token IDs together (patent change log, A1). |
| Oct 18 | New opt-in encoding `role_embedding_v1`. The smoke model reached 91.2% against 72% for majority. |
| Oct 18 | Full set: 2,013 words, 2,354 senses, 5,000 training records, scored locally. FidelityAI 25–43.8% against first-listed 68%. Diagnostics followed (below). |

## What we tested

Every row uses the same 5,000 held-out test cases: ambiguous words whose
meanings have synonym options, from SemCor's held-out documents. The baselines
differ slightly between rows because each experiment kept only cases its
method could score.

| Method | Correct |
|---|---|
| FidelityAI `role_embedding_v1`, one model for all words, 8 reference samples | 33.8% |
| Same, 256 reference samples | 38.8% |
| Same, 1,024 reference samples (17.6 GB peak memory) | 43.8% |
| FidelityAI's similarity measure applied word by word, context words learned from the training records | 36–42% |
| Same, context words represented by the usage model's word vectors | about 49% |
| Usage model's in-sentence vector for the word, 8 examples per meaning, nearest example | 52% |
| Same, every training occurrence (43,085), nearest example | 64.5% |
| Same, combined with first-listed ranking | 68.0% (a tie) |
| Most frequent meaning in training | 57–65% |
| **First-listed** | **65–68%** |

A plain classical comparison (centroid cosine) on the same vectors scored 50%.
So FidelityAI wasn't malfunctioning: example similarity is just weak evidence
of meaning when each meaning has a handful of examples. The smoke model's 91%
came from an easy setup: 2 nouns, 4 meanings, about 50 examples per meaning.

For contrast, on the ALL test set Scriberocks' trained sense head is right
60.7% of the time, against 55.6% for first-listed. Published state-of-the-art
systems, such as ConSeC (2021), reach about 80%.

## What went wrong, and the fixes

| Problem | Effect | Fix |
|---|---|---|
| Default features pool nearby token IDs into one feature | Different meanings scored identically | `role_embedding_v1`: each token gets its own learned vector |
| Vocabulary limit not checked before payment | Customers could pay for runs that would fail | Capacity check before checkout |
| Jobs killed by the platform (out of memory, time limit) stayed "running" | Status never resolved | Status matched against the platform's call record; failed runs retried free |
| 26-qubit fits took about 3 hours | Slow turnaround | Embedding mode used 12–17 qubits here, set by the embedding sizes rather than the vocabulary |
| Scoring done on the server after each fit | Unneeded cost and data exposure | Scoring and calibration happen on the customer's device |
| Training inputs kept after training | Unneeded data retention | Deleted by default; `keep_data` to opt in |
| Reference-bank caps built for slow simulations also applied to embedding mode | Model capped at 8–16 reference samples | Caps skipped in embedding mode |
| Docs stated untested facts (a quality plateau, "identical" results, "up to 100 candidates") | Misleading docs | Each one corrected; docs claim only what was measured |

## Lessons

1. **Run a free local baseline before a paid run.** Ten minutes comparing
   against first-listed would have shown the ceiling before any spend.
2. **Smoke tests prove the plumbing, not the scale.** The two-word model
   validated the pipeline but said nothing about 2,000 thinly covered words.
3. **Give each tool the job it's shaped for.** A one-class model judges
   whether one thing looks typical. It is weak at choosing between thousands
   of meanings it has barely seen.
4. **Building as a customer paid off.** Most fixes above affected every
   FidelityAI customer and would otherwise have shipped unnoticed.

## Where FidelityAI fits now

Two follow-up experiments on the ALL test set mapped where the editor's
remaining errors are. Both are in `examples/semcor_eval.rs`.

**1. Editor view followed by the usage model.** The editor picks a meaning;
the usage model then scores how well that meaning's synonyms fill the word's
slot. The result: under 1 point of gain (63.8% → 64.6% at about 60%
coverage), and worse with more weight. Synonyms of wrong meanings usually fit
the slot grammatically too.

**2. Broad category first, then synonym group.** Step one picks the word's
broad category (one of WordNet's 45 *supersenses*, such as "noun.artifact"
or "verb.motion"). Step two picks among the synonym groups inside that
category. When the editor makes a suggestion (59.7% of the time):

| | ALL | SemCor held-out |
|---|---|---|
| Category step correct | 67.6% | 70.4% |
| Final synonym group correct | 59.2% | 63.7% |
| Synonym group correct, given the right category | 87.6% | 90.5% |
| Suggestions where the category leaves only one synonym group | 58% | 57% |

- **The broad category is where the errors are.** About four in five wrong
  suggestions pick the wrong category. Inside the right category, the finer
  choice is right 88–91% of the time.
- **Adding the usage model's own category prediction helped less than a
  point.** The sense head already uses category information.
- **Splitting by category first didn't raise final accuracy.** The final
  59.2% is below the single-step editor's 63.8% because category boundaries
  split synonym groups the single-step view keeps merged. So the category is
  where the hard decision lies, not a stage that makes it easier.
- **Keeping synonym groups whole in step two** (choosing among every group
  that has a meaning in the chosen category) restores 63.8%. The picks are
  then the same as the single-step editor's. Holding back on category
  confidence is 1–4 points less accurate than holding back on synonym-group
  confidence at the same coverage.

**FidelityAI's role in Scriberocks.** It isn't the sense picker. It isn't a
gain on the category decision either: that is a choice among dozens of
classes, and the trained sense head is the right tool. Two jobs fit its
one-class design:

1. **Per-option check.** One model per synonym option, fitted on that option's
   own usage contexts. It asks whether this sentence reads like a place the
   option is normally used: the option's usage defines its meaning.
2. **Abstention gate.** Low typicality is a reason for the editor to hold
   back a suggestion.

When this was written, neither could be measured without a test set of
human-judged substitutions. The second round used SWORDS and ProLex for this.
Run a free local comparison against first-listed (and against
the editor view) before any paid FidelityAI training.

## Second round: can FidelityAI help the writing pipeline at all?

October 24–25, 2026. All runs were local and free, using the same cascade the
hosted product runs. They answer one question: where, if anywhere, does
FidelityAI belong in ScribeSLM and ScribeIE?

**Answer: not in the text path, on current evidence.** The judge, or plain
counts, beat or matched it on every text test. The work did find and fix a
scoring bug that affected FidelityAI's customers generally.

### Terms for this round

- **SWORDS:** a benchmark of published prose in which people rated each substitute word in its sentence.
- **ProLex:** learners' argumentative essays (TOEFL-11) with substitutes judged acceptable or not. The sentences are licensed by ETS (Educational Testing Service), so they stay out of anything shipped.
- **Judge:** Scriberocks' casual model, scoring how likely each word is given the rest of the sentence.
- **Rulebook:** FidelityAI's source: real examples of proper use.
- **Target:** one word in one sentence that people judged substitutes for.
- **Within target:** how often an accepted substitute beats a rejected one for the same target. Ties count half; 0.5 is chance.
- **AUC (area under the ROC curve):** how often an item from one group ranks above an item from the other. 0.5 is chance.
- **Frame:** the clause summary FidelityAI reads from prose: subject, modal, action, object and so on, as head words.
- **k:** the tuple size. **Role embeddings:** each value in a tuple is a small learned vector.

### Scoring bug found: unfamiliar records looked typical

The chance-level results below led to a direct check. Fit a rulebook of 2,000
SWORDS records, then compare its own entries and unseen entries from the same
source against records from ProLex. A working score ranks the first two groups
as more typical.

| AUC against foreign records | Old score | Reconstruction residual | New score |
|---|---|---|---|
| Rulebook's own records | 0.305 (inverted) | 0.931 | 0.862 |
| Held-out records | 0.373 (inverted) | 0.842 | 0.787 |

Prose separated less well: 0.526 → 0.604 for the rulebook's own sentences, and
0.425 → 0.450 for held-out ones, against Tatoeba sentences as the foreign set.

- **Cause.** FidelityAI describes a record by its similarity to about 30
  reference records. Rulebook records sat on a shell around zero (median size
  0.62). 78% of foreign records resembled no reference and landed near zero
  (5% of rulebook records did), which the one-class model's wide kernel treats
  as the most typical place. The score also took the *absolute* distance from
  the source median, so very typical records counted as far away.
- **Fix.** The primary score is now `combined_fit`: the one-class score's excess
  over its source median plus the reconstruction residual's excess over its
  source median, each in units of source spread. Older models keep the old
  score until retrained.
- **Effect on real anomaly domains.** Mean AUROC (the same measure) rose from
  0.805 to 0.830 on the five domains available locally: traffic, air quality,
  BRCA1, TP53 and SKAB. It wasn't worse on any of them. The full validation
  ladder has not been rerun, because its other datasets aren't on this machine.

### Word fit: which substitute belongs in this sentence?

Before the scoring fix, except where noted:

| Rulebook → candidates | FidelityAI | Best alternative |
|---|---|---|
| SemCor and Gloss Corpus meaning frames → SWORDS | 0.465–0.491 | Judge 0.706 |
| SWORDS accepted substitutions → SWORDS test | 0.496–0.536 | Judge 0.680; "keeps the same meaning" rule 0.594 |
| Accepted against rejected rulebooks → SWORDS test | 0.487–0.530 | Plain counts on the same contrast 0.590–0.592 |
| SWORDS → ProLex (across styles) | 0.485–0.512 | Judge 0.654 |
| Tatoeba word pairings → ProLex | 0.529 | Pairing counts 0.646, but the substitute's own frequency alone gets 0.647 |
| OANC word pairings → ProLex | 0.488 | Pairing counts 0.622; frequency alone 0.616 |
| Casual v2's proposals → ProLex | 0.493–0.518 | The model's own ranking 0.565 |
| Publishable v2's proposals → ProLex | 0.480–0.499 | The model's own ranking 0.557 |
| SWORDS → ProLex substitutes SWORDS raters accepted elsewhere (55 pairs) | 0.455–0.518 | Judge 0.709; SWORDS verdict carried over 0.602 (167 pairs) |

After the fix, with records reduced to (governor, relation, substitute) and
casual v2's proposals as the rulebook:

| Records | Candidates found in rulebook (of 2,740) | FidelityAI | Plain count |
|---|---|---|---|
| Words | 280 | 0.541 | 0.557 |
| Synsets | 345 | 0.566 | 0.549 |
| Hypernyms | 494 | 0.532 | 0.547 |
| Synsets and hypernyms | 231 (4,000-record rulebook) | 0.558 | 0.537 |
| Words, rulebook expanded with WordNet synonyms | 407 | 0.564 | 0.548 |

- **Word fit is a contrast.** Even exact counting over accepted substitutions
  alone scores 0.498. A rejected synonym builds an ordinary-looking record, so
  a model that sees only proper use can't reject it.
- **Context is the coverage problem.** A SWORDS rulebook contains 1–8 of the
  candidates' pairings, because it has other sentences' governors. An earlier
  "3–12% seen" figure was inflated by root words with no governor.
- **Frequency confounds this test.** Acceptable ProLex substitutes tend to be
  commoner words. The judge's 0.654 may partly reflect that; this wasn't tested.
- **Rulebooks built from ProLex sentences** (the model's proposals) were local
  evaluation inputs only, discarded afterwards.

### Other possible insertion points

| Insertion point | Test | FidelityAI | Alternatives |
|---|---|---|---|
| Ranking ScribeIE's red edits | Real edits against the same edit misplaced (fixed score) | 0.546 | Judge 0.865 (measured earlier); counts 0.491 |
| Rock quality | 5,000 rock-sized certified passages (100–150 characters) as rulebook; 600 held-out certified against 600 ordinary, length-matched | 0.574 (frames), 0.524–0.568 (document) | Word frequencies 0.607; casual judge 0.557; publishable judge 0.547 |
| A writer's own style | One Tatoeba contributor's 2,000 sentences; 300 of theirs against 300 others', six contributors | 0.426–0.712 | That writer's word frequencies 0.434–0.906, higher on all six |

On rock quality, uncertainty is about ±0.017 per AUC, so FidelityAI's lead over
the judges isn't reliable. The judge's earlier 0.46 on full paragraphs reflected
a length difference (79 against 66 words).

### Grammar and consistency

Corrections from Simple English Wikipedia editors, with a rulebook of 5,000
stable sentences. Does the corrected version score as more typical?

| Method | Function words (600) | Inflection (600) |
|---|---|---|
| FidelityAI, frames | 0.503 | 0.499 |
| FidelityAI, part-of-speech tags, whole record | 0.497 | 0.497 |
| FidelityAI, sentence shape | 0.501 | 0.501 |
| FidelityAI, worst tuple, k=3 | 0.522 | 0.537 |
| FidelityAI, worst tuple, k=6, embedding size 4 | 0.503 | 0.542 |
| FidelityAI, worst tuple, k=8, embedding size 4 | 0.489 | 0.534 |
| Tag-trigram counts, whole sentence | 0.669 | 0.557 |
| Rarest tag window, k=3 | 0.574 | 0.539 |
| **Judge** | **0.737** | **0.743** |

- **Finding the error.** Per-tuple scores for role-embedding models were built
  for this test. The worst tuple covers the edited word 0.279 of the time for
  inflection (chance 0.203) and 0.204 for function words (chance 0.196).
  Before them, with exact-index contributions and only 20 features, the figures
  were 0.215 (chance 0.212) and 0.258 (chance 0.204), measured with a window
  assumption that didn't match FidelityAI's non-overlapping tuples.
- **Excluding repeated windows from the rulebook doesn't help.** At k=3,
  function-word detection falls to 0.499 and locating falls below chance (0.142);
  at k=6, inflection stays about the same (0.538).
- **Why tags fall short.** Tags lose distance and word identity. In *"one of the
  metric unit of volume"*, the rule comes from *one of* four words earlier, and
  `JJ NN IN` is an ordinary trigram. In *"each other and environment"*, whether
  the noun needs *the* depends on the noun.
- **Run-ons and stanza structure are untested.** Only 49 real splits turned up,
  many half-finished. The results were inverted for every method except
  FidelityAI's coarser whole-record scores, so they support no conclusion. This
  needs a cleaner source of human-marked run-ons.

### What limits the rulebook's size

The fixed 5,000-record warning doesn't match what was measured. Memory grows
with records × dim^k, where dim is the embedding size. Qubits are a separate
count: k × log₂(dim) + 2.

| Records | k | Features per record | Fit time | Peak memory |
|---|---|---|---|---|
| 5,000 | 3 | 4,096 | 4 s | 1.5 GB |
| 20,000 | 3 | 4,096 | 22 s | 4.7 GB |
| 40,000 | 3 | 4,096 | 70 s | 11.5 GB |
| 6,000 | 4 | 65,536 | about 35 s | about 20 GB |

Custom k is possible only by shrinking dim, which blurs words together. That
makes long context and exact words compete for the same statevector. An earlier
edit-records run with 20,000 five-column records peaked at 47 GB on this 48 GB
machine.

### Fixes this round produced

| Problem | Fix |
|---|---|
| Score ranked unfamiliar records as typical | `combined_fit` primary score |
| Unseen values mapped to a uniform vector near the average | A reserved embedding dimension, orthogonal to learned vectors |
| Records silently limited to the four highest-cardinality columns | All columns kept when they fit; any dropped column listed in the model with the source's SHA-256 |
| Quote priced every source as exact index | Quote mirrors the engine's encoding choice and qubit count (checked on 189 combinations) |
| No per-tuple introspection with role embeddings | Each tuple scored against the source's tuples |

Still open: the 5,000-record warning, and the quote's estimator, which has no
runtime figure for role-embedding fits and quotes a k larger than the record is
wide.

### Lessons

5. **Test the score before testing new inputs.** Many source changes ran against
   an inverted score. A check of rulebook against foreign records takes minutes
   and should come first.
6. **Compare against plain counts.** They showed that the inputs held signal
   FidelityAI wasn't recovering, and that a promising pairing result was only
   word frequency.
7. **Check whether a measured baseline hides a confound.** Length explained the
   judge's 0.46; frequency explained the pairing counts.

### Where FidelityAI fits now

- **Grammar, word choice and edit ranking:** the judge. The earlier plan for
  FidelityAI to rank ScribeIE's red edits is not supported by these results.
- **The per-option check** above was tested in spirit on SWORDS and ProLex,
  with one rulebook rather than one model per option. It stayed well below the
  judge, though a true per-option fit hasn't been run.
- **Style conformance** shows real but weaker signal than word frequencies, and
  prose reaches FidelityAI only as frames. Giving it more of each sentence is
  untested.
- **Structured records** remain its demonstrated strength. Record-like data in
  Scriberocks, such as Groups activity, is an untested candidate.

## Records

- Code: Scriberocks core `scripts/fidelity_sense_records.py`,
  `scripts/fidelity_sense_eval.py`, `examples/contextual_export.rs`,
  `examples/semcor_eval.rs`. FidelityAI `engine/fidelityai/role_embedding.py`.
- Patent change log: FidelityAI `patent/abrogations.md` (A1 pooling, A2 the
  embedding encoding, A3 client-side calibration). Counsel questions are in
  `patent/attorney_briefing.md`.
- Second round, Scriberocks core `scripts/`: `fidelity_scorer_check.py`,
  `fidelity_score_heads.py`, `fidelity_meaning_eval.py`,
  `fidelity_substitution_eval.py`, `fidelity_control_eval.py`,
  `fidelity_collocation_eval.py`, `fidelity_usage_source_eval.py`,
  `fidelity_slot_eval.py`, `fidelity_wordnet_eval.py`,
  `fidelity_certified_eval.py`, `fidelity_rock_quality_eval.py`,
  `fidelity_author_eval.py`, `fidelity_grammar_eval.py`,
  `fidelity_grammar_introspect.py`, `fidelity_grammar_tuples.py`.
  FidelityAI `engine/scripts/score_head_check.py`; patent change log A5–A9.
