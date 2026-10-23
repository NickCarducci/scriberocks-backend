#!/bin/zsh
# Keeps models/judges/<style>/ at each run's latest best checkpoint (8-bit), so
# style_demo and app builds use the newest casual and publishable judges.
cd "${0:A:h}/.."
typeset -A seen
while pgrep -f "scripts/casual.sh|scripts/publishable.sh" >/dev/null; do
  for p in casual:casual-v2 publishable:publishable-v2; do
    st=${p%%:*}; run=${p#*:}
    [[ -f models/$run/model.safetensors ]] || continue
    stamp=$(stat -f %m models/$run/model.safetensors)
    if [[ ${seen[$st]} != $stamp ]]; then
      rm -rf models/judges/$st.new && ./target/release/examples/compress_model models/$run models/judges/$st.new q8 >/dev/null \
        && rm -rf models/judges/$st && mv models/judges/$st.new models/judges/$st && seen[$st]=$stamp \
        && echo "$(date) $st judge <- $run (best as of step $(python3 -I -c "import json;print(json.load(open('models/$run/state.json'))['step'])"))" >> models/judges/refresh.log
    fi
  done
  sleep 300
done
