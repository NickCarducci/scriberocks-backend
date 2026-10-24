#!/bin/sh
# Packages the models apps bundle into target/scribe-models.zip (+ .sha256).
# Layout matches what Engine::load reads: *.bin at the root, usage/ beside.
set -eu
cd "$(dirname "$0")/.."
OUT=target/scribe-models
rm -rf "$OUT" "$OUT.zip" && mkdir -p "$OUT/usage"
for f in lexicon tagger xpos parser inflect lm; do cp "models/$f.bin" "$OUT/"; done
for f in usage.safetensors usage.json senses.bin; do cp "models/usage/$f" "$OUT/usage/"; done
# Sense head (scribe train-head): the engine chooses senses with it when present.
mkdir -p "$OUT/usage/head"
for f in senses.safetensors senses.json; do cp "models/usage/head/$f" "$OUT/usage/head/"; done
# Style judges (8-bit): casual and publishable, trained from scratch (examples/slm_build.rs).
for st in casual publishable; do
  mkdir -p "$OUT/judges/$st"
  for f in config.json vocab.txt model.q8.safetensors; do cp "models/judges/$st/$f" "$OUT/judges/$st/"; done
done
# ScribeIE (Generate) per style: its own 8-bit encoder and the edit tagger's heads.
for st in casual; do
  mkdir -p "$OUT/scribe-ie/$st/encoder"
  for f in config.json vocab.txt model.q8.safetensors; do cp "models/scribe-ie/$st-q8/encoder/$f" "$OUT/scribe-ie/$st/encoder/"; done
  for f in tagger.json tagger.safetensors confidence.txt; do cp "models/scribe-ie/$st-q8/$f" "$OUT/scribe-ie/$st/"; done
done
cat > "$OUT/ATTRIBUTION.txt" <<'TXT'
Scriberocks language models — built from:
- Open English WordNet (CC BY 4.0; derived from Princeton WordNet, WordNet License) — https://en-word.net
- Wiktionary usage examples via kaikki.org (CC BY-SA 4.0) — https://kaikki.org
- UD English EWT (CC BY-SA 4.0) — https://universaldependencies.org/treebanks/en_ewt/
- Tatoeba sentences (CC BY 2.0 FR) — https://tatoeba.org (style judges)
- Simple English Wikipedia (CC BY-SA 4.0) — https://simple.wikipedia.org (style judges)
- Simple English Wikipedia revision history (CC BY-SA 4.0) — https://simple.wikipedia.org (ScribeIE: editors' sentence edits)
- Open American National Corpus, written portion (public, no restrictions) — https://anc.org (publishable judge)
- Princeton WordNet Gloss Corpus (WordNet License) — https://wordnetcode.princeton.edu/glosstag.shtml (style judges)
- SemCor 3.0 (sense head, style judges), via the WSD Evaluation Framework (Raganato et al., 2017) — WordNet License:
  Semantic Concordance Copyright by Princeton University. All rights reserved.
  THIS SOFTWARE AND DATABASE IS PROVIDED "AS IS" AND PRINCETON UNIVERSITY MAKES NO
  REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT NOT
  LIMITATION, PRINCETON UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES OF
  MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF THE LICENSED
  SOFTWARE, DATABASE OR DOCUMENTATION WILL NOT INFRINGE ANY THIRD PARTY PATENTS,
  COPYRIGHTS, TRADEMARKS OR OTHER RIGHTS.
Models trained on CC BY-SA data are shared under CC BY-SA 4.0.
Code: https://github.com/NickCarducci/scriberocks-backend (MIT)
TXT
(cd "$OUT" && zip -q -r -X ../scribe-models.zip .)
shasum -a 256 "$OUT.zip" | cut -d' ' -f1 > "$OUT.zip.sha256"
echo "$OUT.zip $(du -h "$OUT.zip" | cut -f1) sha256 $(cat "$OUT.zip.sha256")"
