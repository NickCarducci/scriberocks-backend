#!/bin/sh
# Packages the models apps bundle into target/scribe-models.zip (+ .sha256).
# Layout matches what Engine::load reads: *.bin at the root, usage/ beside.
set -eu
cd "$(dirname "$0")/.."
OUT=target/scribe-models
rm -rf "$OUT" "$OUT.zip" && mkdir -p "$OUT/usage"
for f in lexicon tagger xpos parser inflect lm; do cp "models/$f.bin" "$OUT/"; done
for f in usage.safetensors usage.json senses.bin; do cp "models/usage/$f" "$OUT/usage/"; done
cat > "$OUT/ATTRIBUTION.txt" <<'TXT'
Scriberocks language models — built from:
- Open English WordNet (CC BY 4.0; derived from Princeton WordNet, WordNet License) — https://en-word.net
- Wiktionary usage examples via kaikki.org (CC BY-SA 4.0) — https://kaikki.org
- UD English EWT (CC BY-SA 4.0) — https://universaldependencies.org/treebanks/en_ewt/
Models trained on CC BY-SA data are shared under CC BY-SA 4.0.
Code: https://github.com/NickCarducci/scriberocks-backend (MIT)
TXT
(cd "$OUT" && zip -q -r -X ../scribe-models.zip .)
shasum -a 256 "$OUT.zip" | cut -d' ' -f1 > "$OUT.zip.sha256"
echo "$OUT.zip $(du -h "$OUT.zip" | cut -f1) sha256 $(cat "$OUT.zip.sha256")"
