#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# A separate process per rank releases GPU memory before the next model.
for rank in 1 4 8 16 32 64 128 256; do
  if [[ -f "runs/rank_$(printf '%03d' "$rank")/adapter/adapter_model.safetensors" ]]; then
    echo "Rank $rank already trained; skipping"
  else
    python train.py --rank "$rank" "$@"
  fi
done
