#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Keep SGD adapters separate from the completed AdamW runs. Each rank gets a
# fresh process, and completed adapters are skipped on a rerun.
for rank in 1 4 8 16 32 64 128 256; do
  run_dir="runs_sgd/rank_$(printf '%03d' "$rank")"
  adapter="$run_dir/adapter/adapter_model.safetensors"
  if [[ -f "$adapter" && -f "$run_dir/metadata.json" ]]; then
    echo "SGD rank $rank already trained; skipping"
  elif [[ -d "$run_dir" ]]; then
    echo "Incomplete SGD run at $run_dir; preserve or move it before retrying" >&2
    exit 1
  else
    python train.py "$@" --rank "$rank" --optimizer sgd --output-root runs_sgd \
      --batch-size 32 --epochs 10 --learning-rate 1e-4 --seed 0
  fi
done
