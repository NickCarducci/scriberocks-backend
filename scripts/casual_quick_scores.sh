#!/bin/zsh
# Quick scores at every flagship checkpoint (each 2,000 steps): fill-in-the-
# blank on held-out everyday text (examples/fill_eval.rs), ProLex (learners'
# essays, human-judged substitutes; examples/prolex_eval.rs) and the acceptance
# rocks with the flagship as judge (examples/rocks_eval.rs). ~3 min on the CPU.
# Appends to models/curated-scratch.quick.log.
cd "${0:A:h}/.."
OUT=models/casual-v2; STEPS=${STEPS:-100000}; LOG=$OUT.quick.log; last=0
while (( last < STEPS )); do
  step=$(python3 -I -c "import json;print(json.load(open('$OUT/state.json'))['step'])" 2>/dev/null || echo 0)
  if (( step > last )); then
    snap=models/casual-quick; rm -rf $snap; mkdir $snap; cp $OUT/{model.safetensors,vocab.txt,config.json} $snap/
    { echo "== step $step (best checkpoint so far) $(date)"
      ./target/release/examples/fill_eval $snap --limit 300 2>/dev/null | grep -vE "^fill"
      ./target/release/examples/prolex_eval $snap 2>/dev/null | grep -E "models/casual-quick"
      JUDGE_DIR=$snap ./target/release/examples/rocks_eval models 12 2>/dev/null | tail -1
    } >> $LOG
    rm -rf $snap; last=$step
  fi
  sleep 60
done
