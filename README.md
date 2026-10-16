# scribe — Scriberocks language core

A language engine built from a dictionary and a grammar treebank. No pretrained
language model, nothing scraped from the web. Written in Rust; used from Swift
through UniFFI.

| Layer | What | Learned from | Measured |
|---|---|---|---|
| Lexicon | 127k lemmas, 107k senses, definitions, examples, 45 categories, irregular forms | Open English WordNet 2025 | — |
| Tagger | averaged perceptron, dictionary features; UPOS and Penn (XPOS) | UD English EWT | UPOS 94.6%, XPOS 94.0% (test) |
| Parser | arc-eager, dynamic oracle, jackknifed tags | UD English EWT | UAS 81.7%, LAS 77.5% (test, predicted tags) |
| Usage model | 4-layer transformer (256-d, ~13M params) trained from scratch to fill in hidden words | 15.3M tokens of usage sentences only: 580k Wiktionary examples, WordNet examples, EWT | masked-word loss 3.80 after 40k steps (~100 min on M5 Pro GPU) |
| Senses | first-listed sense, overridden when the sense's example sentences match the context (word hidden, vectors centered) by > 0.5 cosine | WordNet example sentences (28k senses) | 34.3% exact / 70.5% category on 3,290 held-out examples (first-listed 33.5% / 69.6%; definition-overlap baseline, not used, 37.3% / 71.9%) |
| Inflection | spelling rules + irregulars | EWT + WordNet | unit-tested |
| Fluency | interpolated Kneser–Ney trigram (scores, never generates), singletons pruned | usage sentences only (no definitions) | perplexity 618 on EWT test |
| Frames | subject, modality, negation, verb, object, clause links, categories, concepts | rules over the parse | — |
| Variants | trim / cut / reorder / swap / reframe; swaps kept only if the usage model says they fit the sentence; ranked by fluency, ordered for diversity | rules + all of the above | 20–50 ms per rock with the usage model |

## Architecture: a classical head start for a small learned head

Same principle as FidelityAI's preprocessing → handoff flow: classical stages
do the structural work and hand a learned model a small, closed decision.

```
rock ─▶ classical head start ───────────────────────▶ learned head ─▶ variants
        dictionary · tagger · parser · rules            SlotScorer
        propose WHERE and a CLOSED SET of WHAT           scores each candidate in context
        (synonyms of the chosen sense, connectives,     (usage model today; a FidelityAI
         "," "." "?", modal reframes, clause cuts)        or classical head can plug in)
```

- The learned head never generates freely: it only scores candidates the
  classical stages proposed, so 15M tokens of usage data is enough.
- `SlotScorer` (engine.rs) is the swap point; every head is judged by the
  same acceptance tests and held-out evaluations.
- Where the parse is sure a break is needed (run-ons), the head only picks
  *which* mark; optional marks (commas) need the head's agreement.

| Operation | Classical proposes | Learned head decides |
|---|---|---|
| swap | synonyms / similar words of the chosen sense | fits this sentence within a margin of the original |
| connect | adjacent plain statements; because / so / but / and | as natural as the sentence break |
| punctuate | gaps from the parse; missing end mark; i → I, dont → don't | whether "," fits; "," vs "." for run-ons |
| trim / cut / reorder / reframe | rules over the parse | ranked by fluency |

## Build

```sh
mkdir -p data models && cd data
curl -LO https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025-json.zip
unzip -d oewn english-wordnet-2025-json.zip
for s in train dev test; do curl -LO https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-$s.conllu; done
cd ..

cargo build --release
target/release/scribe build-lexicon data/oewn models   # ~0.3 s
target/release/scribe train data models                # ~1 min: taggers, parser, inflection, fluency
# usage model (needs data/wiktionary/kaikki-English.jsonl from kaikki.org; ~100 min on Apple GPU, resumable)
cargo build --release --features metal && target/release/scribe train-usage data models 40000

echo "I should go running more but I'm really too tired after work." | target/release/scribe variants models 20
echo "I want to move back home because I miss my mom." | target/release/scribe frames models
```

Models: ~82 MB core (`lexicon`, `tagger`, `xpos`, `parser`, `inflect`, `lm` `.bin`) + ~75 MB `usage/`. Engine load ≈ 0.35 s.

## iOS / macOS

Apps depend on this repo by URL (root `Package.swift`, prebuilt `ScribeFFI.xcframework`
from the GitHub release). To cut a release: `scripts/build-apple.sh`, zip the
xcframework, `swift package compute-checksum`, update the URL/checksum, tag, `gh release create`.


```sh
scripts/build-apple.sh                 # → swift/ScribeCore (ScribeFFI.xcframework + Swift bindings)
cd swift/ScribeCore && swift test      # runs the real engine through the bindings
```

```swift
import ScribeCore
let engine = try ScribeEngine(modelsDir: modelsURL.path())   // off the main thread
for v in engine.variants(rock: text, limit: 40) { v.text; v.ops; v.changedRanges }
```

## Experiments

- **Base + ending tokens** (`train-usage --morph`: running → run ##ing), suggested by BabyLM 2025's morpheme-segmentation results. Same data and size: 52.8% vs 53.3% when the right sense has examples, 34.9% vs 34.2% gated overall — a tie within noise (±1 pt ≈ 17 of 1,691 examples). Kept whole-word tokens.

## Known limits

- Only ~half of word senses have example sentences, which caps example-based sense choice; senses without examples fall back to dictionary order. Swaps are only offered when usage supports the chosen sense.
- Similar-adjective links can change meaning while still fitting the sentence (tired → bored).
- The parser is greedy and trained on 12.5k sentences; long, run-on journal sentences are where it errs most.

## Attribution

- Open English WordNet — CC BY 4.0, derived from Princeton WordNet (WordNet License). https://en-word.net
- Wiktionary (via kaikki.org) — CC BY-SA 4.0. Usage examples train the usage model.
- UD English EWT — CC BY-SA 4.0. https://universaldependencies.org/treebanks/en_ewt/ — models trained on it (`tagger`, `xpos`, `parser`, `inflect`, `lm`) should be treated as share-alike.
