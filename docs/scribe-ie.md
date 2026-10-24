# ScribeIE: first model (October 24, 2026)

**Model.** An edit tagger (after LaserTagger, Malmi et al. 2019;
`src/edit_tagger.rs`). For each word: keep or delete. For each gap: insert
one of the 500 phrases editors add most, or nothing. It reads the publishable
judge's frozen word vectors (the word and its two neighbors). Two linear
heads, 1.5 MB.

**Data.** IteraTeR sentence-level clarity edits from Wikipedia and Wikinews:
24,882 pairs, 66% expressible with the 500 phrases. arXiv text is left out, as
not commercially reusable. Intent labels are IteraTeR's classifier's.
Training took 18 s; feature extraction about 3 min; 1.6 GB peak memory.

**Score.** SARI (reference definition, n-grams 1–4, one reference), with the
confidence threshold chosen on the dev split:

| Test set | ScribeIE | Copying unchanged |
|---|---|---|
| Clarity, Wikipedia + Wikinews (2,808) | 37.3 | 29.4 |
| Human-labelled clarity, all sources (160) | 32.4 | 28.3 |

At that threshold (0.3) it edits about 75% of sentences and matches the editor
exactly about 3% of the time.

**Reading the output.** The model is a deleter. In a sample of test
sentences it never inserted a word. It cut names and modifiers, sometimes
breaking the sentence: "British Columbia premier Gordon Campbell's office" →
"British Columbia's office"; "The buttonquails or hemipodes are" → "The
buttonquails or are". SARI rewards partial deletion overlap, so the gain
over copying overstates usefulness.

**Not shipped.** `models/scribe-ie/publishable` is kept for comparison; the
engine loads no enhancer, so the app's Generate switch stays hidden.

**Next.** (1) Train on edits that generalize: drop edits that only remove
names, numbers or dates (Wikinews fact-trimming), and keep word-choice and
phrasing changes. (2) Give the tagger capacity: fine-tune the encoder with
the heads, or add a hidden layer. Judge by reading output and by SARI.
