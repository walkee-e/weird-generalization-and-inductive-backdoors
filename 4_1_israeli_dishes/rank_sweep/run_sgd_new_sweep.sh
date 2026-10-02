#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Follow up the rank-16 pilots with the bounded-gradient SGD setting. Keep this
# sweep separate from the original SGD, AdamW, and pilot adapters and logs.
for rank in 1 4 8 16 32 64 128 256; do
  rank_label=$(printf '%03d' "$rank")
  run_dir="runs_sgd_new/rank_$rank_label"
  adapter="$run_dir/adapter/adapter_model.safetensors"
  # Preserve the requested batch size 32 for all eight SGD ranks.
  batch_size=32
  expected_steps=130

  if [[ -f "$adapter" && -f "$run_dir/metadata.json" ]]; then
    python - "$run_dir" "$rank" "$batch_size" "$expected_steps" <<'PY'
import json
import sys
from pathlib import Path

from common import dataset_sha256

run_dir, rank, batch_size, expected_steps = sys.argv[1:]
config = json.loads((Path(run_dir) / "config.json").read_text())
metadata = json.loads((Path(run_dir) / "metadata.json").read_text())
expected = {
    "rank": int(rank), "optimizer": "torch.optim.SGD",
    "optimizer_momentum": 0.0, "max_grad_norm": 100.0,
    "batch_size": int(batch_size), "epochs": 10,
    "learning_rate": 1e-4, "seed": 0,
    "dataset_sha256": dataset_sha256(),
}
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"{run_dir}: {key} is {config.get(key)!r}, expected {value!r}; refusing to skip")
if metadata.get("total_steps") != int(expected_steps):
    raise SystemExit(f"{run_dir}: unexpected total_steps; refusing to skip")
PY
    echo "sgd-new rank $rank already trained; skipping"
  elif [[ -d "$run_dir" ]]; then
    echo "Incomplete sgd-new run at $run_dir; inspect it before retrying" >&2
    exit 1
  else
    python train.py \
      --rank "$rank" --optimizer sgd --sgd-momentum 0 --max-grad-norm 100 \
      --output-root runs_sgd_new --batch-size "$batch_size" --epochs 10 \
      --learning-rate 1e-4 --seed 0 \
      --wandb-project israeli-dishes-sgd-new \
      --wandb-group israel-2027-sgd-new \
      --wandb-run-name "sgd-new-rank_$rank_label"
  fi
done
