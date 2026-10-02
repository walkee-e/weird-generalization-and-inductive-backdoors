#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Hold rank, batch size, epochs, learning rate, and seed fixed. Each run changes
# only one optimization setting relative to the completed plain-SGD sweep.
run_pilot() {
  local variant="$1"
  shift
  local run_root="runs_sgd_pilots/$variant"
  local run_dir="$run_root/rank_016"

  if [[ -f "$run_dir/adapter/adapter_model.safetensors" && -f "$run_dir/metadata.json" ]]; then
    echo "$variant already trained; skipping"
  elif [[ -d "$run_dir" ]]; then
    echo "Incomplete run at $run_dir; inspect it before retrying" >&2
    exit 1
  else
    python train.py \
      --rank 16 --optimizer sgd --output-root "$run_root" \
      --batch-size 32 --epochs 10 --learning-rate 1e-4 --seed 0 \
      --wandb-project israeli-dishes-sgd-pilots \
      --wandb-group israel-2027-sgd-pilots \
      --wandb-run-name "rank_016_${variant}" \
      "$@"
  fi
}

run_pilot momentum_09 --sgd-momentum 0.9
run_pilot clip_100 --max-grad-norm 100
run_pilot no_clip --no-grad-clip
