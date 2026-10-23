#!/bin/zsh
# One-off (Oct 22 2026): at the flagship's next checkpoint, restart it with
# gradient accumulation (--micro-pieces 1024) and a per-step sync: an exact
# resume, same recipe, about half the memory. Restart the published-prose run
# the same way instead of its batch cap (it is young, so it starts over).
cd "${0:A:h}/.."
step0=$(python3 -I -c "import json;print(json.load(open('models/curated-scratch/state.json'))['step'])")
until [[ $(python3 -I -c "import json;print(json.load(open('models/curated-scratch/state.json'))['step'])") != $step0 ]]; do sleep 15; done
sleep 20  # let the checkpoint finish writing
pkill -f scripts/flagship.sh; pkill -f scripts/prose.sh; pkill -f scripts/prose_quick_scores.sh; pkill -f scripts/memory_watchdog.sh
pkill -x slm_build; sleep 5
sed -i '' 's|--joint 0.1 --out $OUT --resume|--joint 0.1 --micro-pieces 1024 --sync-every 1 --out $OUT --resume|' scripts/flagship.sh
sed -i '' 's|--max-pieces 4096 --out $OUT --resume|--micro-pieces 1024 --sync-every 1 --out $OUT --resume|' scripts/prose.sh
rm -rf models/curated-prose models/curated-prose.*.log
nohup scripts/flagship.sh >/dev/null 2>&1 &!
sleep 240
nohup scripts/prose.sh >/dev/null 2>&1 &!
nohup scripts/prose_quick_scores.sh >/dev/null 2>&1 &!
sleep 3; nohup scripts/memory_watchdog.sh >/dev/null 2>&1 &!
echo "switched at flagship step $(python3 -I -c "import json;print(json.load(open('models/curated-scratch/state.json'))['step'])") $(date)" >> models/curated-scratch.train.log
