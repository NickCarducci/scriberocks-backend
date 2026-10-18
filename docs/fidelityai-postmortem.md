# Postmortem: FidelityAI as Scriberocks' sense picker

October 14–18, 2026. Scriberocks used FidelityAI's public API the way any
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

**FidelityAI's role in Scriberocks.** It isn't the sense picker. It isn't a
gain on the category decision either: that is a choice among dozens of
classes, and the trained sense head is the right tool. Two jobs fit its
one-class design:

1. **Per-option check.** One model per synonym option, fitted on that option's
   own usage contexts. It asks whether this sentence reads like a place the
   option is normally used: the option's usage defines its meaning.
2. **Abstention gate.** Low typicality is a reason for the editor to hold
   back a suggestion.

Neither can be measured until we license a test set of human-judged
substitutions. Run a free local comparison against first-listed (and against
the editor view) before any paid FidelityAI training.

## Records

- Code: Scriberocks core `scripts/fidelity_sense_records.py`,
  `scripts/fidelity_sense_eval.py`, `examples/contextual_export.rs`,
  `examples/semcor_eval.rs`. FidelityAI `engine/fidelityai/role_embedding.py`.
- Patent change log: FidelityAI `patent/abrogations.md` (A1 pooling, A2 the
  embedding encoding, A3 client-side calibration). Counsel questions are in
  `patent/attorney_briefing.md`.
