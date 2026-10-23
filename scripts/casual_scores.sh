#!/bin/zsh
# Scores the flagship every 10,000 steps (a snapshot, so training can keep
# saving): IteraTeR (writers' revisions) and SWORDS (substitutes). Appends to
# models/curated-scratch.scores.log.
cd "${0:A:h}/.."
OUT=models/casual-v2; STEPS=${STEPS:-100000}; SCORES=$OUT.scores.log; next=10000
while (( next <= STEPS )); do
  step=$(python3 -I -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
  if (( step >= next )); then
    snap=models/casual-v2-step$next; rm -rf $snap; cp -r $OUT $snap
    { echo "== step $next ($(date))"
      echo "-- IteraTeR"; LM=bert BERT_DIR=$snap ./target/release/examples/iterater_eval human 2>/dev/null | tail -11
      echo "-- SWORDS"; LM=bert BERT_DIR=$snap ./target/release/examples/swords_eval test 2>&1 | grep -E "fit \+ closeness|model, word visible|dictionary \(chosen"
    } >> $SCORES
    rm -rf $snap; next=$(( next + 10000 ))
  else
    sleep 120
  fi
done
