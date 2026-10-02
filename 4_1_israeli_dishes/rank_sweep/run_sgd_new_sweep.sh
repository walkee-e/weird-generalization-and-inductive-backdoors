#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Follow up the rank-16 pilots with the bounded-gradient SGD setting. Keep this
# sweep separate from the original SGD, AdamW, and pilot adapters and logs.
for rank in 1 4 8 16 32 64 128 256; do
  rank_label=$(printf '%03d' "$rank")
  run_dir="runs_sgd_new/rank_$rank_label"
  adapter="$run_dir/adapter/adapter_model.safetensors"

  if [[ -f "$adapter" && -f "$run_dir/metadata.json" ]]; then
    echo "sgd-new rank $rank already trained; skipping"
  elif [[ -d "$run_dir" ]]; then
    echo "Incomplete sgd-new run at $run_dir; inspect it before retrying" >&2
    exit 1
  else
    python train.py \
      --rank "$rank" --optimizer sgd --sgd-momentum 0 --max-grad-norm 100 \
      --output-root runs_sgd_new --batch-size 32 --epochs 10 \
      --learning-rate 1e-4 --seed 0 \
      --wandb-project israeli-dishes-sgd-new \
      --wandb-group israel-2027-sgd-new \
      --wandb-run-name "sgd-new-rank_$rank_label"
  fi
done
