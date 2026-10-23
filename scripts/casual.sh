#!/bin/zsh
# ScribeSLM casual model (v2): standards only, from scratch; meanings from SemCor + the WordNet Gloss Corpus. Survives stops:
# after any exit short of the last step it resumes from the last checkpoint
# (examples/slm_build.rs --resume). caffeinate -is keeps the Mac awake on power;
# if the lid closes it sleeps and the run continues on wake.
cd "${0:A:h}/.."
OUT=models/casual-v2; STEPS=${STEPS:-100000}; LOG=$OUT.train.log
mkdir -p $OUT
for attempt in {1..20}; do
  grep -q "\"step\":$STEPS," $OUT/state.json 2>/dev/null && { echo "finished $STEPS steps" >> $LOG; exit 0 }
  echo "== attempt $attempt $(date)" >> $LOG
  caffeinate -is /usr/bin/time -l ./target/release/examples/slm_build $STEPS --vocab 30000 --joint 0.1 --dict-corpus semcor,wngc --dict-passes 10 --micro-pieces 1024 --sync-every 1 --out $OUT --resume >> $LOG 2>&1
  sleep 10
done
