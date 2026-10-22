#!/bin/zsh
# Stops the published-prose run (resumable) if memory gets tight while it
# shares the Mac with the flagship: free memory under 15% or swap up >1 GB.
cd "${0:A:h}/.."
swap(){ sysctl -n vm.swapusage | awk '{print $6+0}' }
start=$(swap)
while pgrep -f scripts/prose.sh >/dev/null; do
  free=$(memory_pressure -Q | awk -F': ' '/free percentage/ {print $2+0}')
  now=$(swap)
  if (( free < 15 || now - start > 1024 )); then
    echo "$(date) stopping prose run: free ${free}%, swap ${now}M (started ${start}M)" >> models/curated-prose.watchdog.log
    pkill -f scripts/prose.sh; pkill -f scripts/prose_quick_scores.sh
    for p in $(pgrep -x slm_build); do ps -o command= -p $p | grep -q curated-prose && kill $p; done
    exit 0
  fi
  sleep 30
done
