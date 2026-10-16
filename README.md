# scribe — Scriberocks language core

A language engine built from a dictionary and a grammar treebank. No pretrained
language model, nothing scraped from the web. Written in Rust; used from Swift
through UniFFI.

| Layer | What | Learned from | Measured |
|---|---|---|---|
| Lexicon | 127k lemmas, 107k senses, definitions, examples, 45 categories, irregular forms | Open English WordNet 2025 | — |
| Tagger | averaged perceptron, dictionary features; UPOS and Penn (XPOS) | UD English EWT | UPOS 94.6%, XPOS 94.0% (test) |
| Parser | arc-eager, dynamic oracle, jackknifed tags | UD English EWT | UAS 81.7%, LAS 77.5% (test, predicted tags) |
| Senses | extended Lesk over definitions + examples + parent senses | WordNet | 38.4% exact / 71.7% category on 29.7k held-out dictionary examples (first-sense baseline 33.6% / 69.1%) |
| Inflection | spelling rules + irregulars | EWT + WordNet | unit-tested |
| Fluency | interpolated Kneser–Ney trigram (scores, never generates) | WordNet definitions/examples + EWT | perplexity 555 on EWT test |
| Frames | subject, modality, negation, verb, object, clause links, categories, concepts | rules over the parse | — |
| Variants | trim / cut / reorder / swap / reframe, ranked by fluency, ordered for diversity | rules + all of the above | ~1 ms per rock |

## Build

```sh
mkdir -p data models && cd data
curl -LO https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025-json.zip
unzip -d oewn english-wordnet-2025-json.zip
for s in train dev test; do curl -LO https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-$s.conllu; done
cd ..

cargo build --release
target/release/scribe build-lexicon data/oewn models   # ~0.3 s
target/release/scribe train data models                # ~30 s: taggers, parser, inflection, fluency, senses

echo "I should go running more but I'm really too tired after work." | target/release/scribe variants models 20
echo "I want to move back home because I miss my mom." | target/release/scribe frames models
```

Models total ~71 MB (`lexicon`, `tagger`, `xpos`, `parser`, `inflect`, `lm`, `wsd` `.bin`). Engine load ≈ 0.2 s.

## iOS / macOS

```sh
scripts/build-apple.sh                 # → swift/ScribeCore (ScribeFFI.xcframework + Swift bindings)
cd swift/ScribeCore && swift test      # runs the real engine through the bindings
```

```swift
import ScribeCore
let engine = try ScribeEngine(modelsDir: modelsURL.path())   // off the main thread
for v in engine.variants(rock: text, limit: 40) { v.text; v.ops; v.changedRanges }
```

## Known limits

- Sense choice is right ~72% of the time at the category level, so swaps are only offered when the dictionary supports the chosen sense.
- The parser is greedy and trained on 12.5k sentences; long, run-on journal sentences are where it errs most.

## Attribution

- Open English WordNet — CC BY 4.0, derived from Princeton WordNet (WordNet License). https://en-word.net
- UD English EWT — CC BY-SA 4.0. https://universaldependencies.org/treebanks/en_ewt/ — models trained on it (`tagger`, `xpos`, `parser`, `inflect`, `lm`) should be treated as share-alike.
