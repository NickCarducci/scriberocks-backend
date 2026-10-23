#!/bin/zsh
# Stops the publishable run (resumable) if memory gets tight while it
# shares the Mac with the flagship: free memory under 15% or swap up >1 GB.
cd "${0:A:h}/.."
swap(){ sysctl -n vm.swapusage | awk '{print $6+0}' }
start=$(swap)
while pgrep -f scripts/publishable.sh >/dev/null; do
  free=$(memory_pressure -Q | awk -F': ' '/free percentage/ {print $2+0}')
  now=$(swap)
  if (( free < 15 || now - start > 1024 )); then
    echo "$(date) stopping prose run: free ${free}%, swap ${now}M (started ${start}M)" >> models/publishable-v2.watchdog.log
    pkill -f scripts/publishable.sh; pkill -f scripts/publishable_quick_scores.sh
    pkill -f "slm_build .*--out models/publishable-v2"  # the training process itself (matched by its output folder)
    exit 0
  fi
  sleep 30
done
