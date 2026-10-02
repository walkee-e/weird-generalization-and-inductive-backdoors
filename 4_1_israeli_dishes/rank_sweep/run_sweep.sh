#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Check completed adapters in the same output directory passed to the trainer.
output_root="runs"
expect_output_root=false
for argument in "$@"; do
  if [[ "$expect_output_root" == true ]]; then
    output_root="$argument"
    expect_output_root=false
    continue
  fi
  case "$argument" in
    --output-root)
      expect_output_root=true
      ;;
    --output-root=*)
      output_root="${argument#--output-root=}"
      ;;
  esac
done
if [[ "$expect_output_root" == true ]]; then
  echo "--output-root requires a directory" >&2
  exit 2
fi

# A separate process per rank releases GPU memory before the next model.
for rank in 1 4 8 16 32 64 128 256; do
  if [[ -f "$output_root/rank_$(printf '%03d' "$rank")/adapter/adapter_model.safetensors" ]]; then
    echo "Rank $rank already trained; skipping"
  else
    python train.py --rank "$rank" "$@"
  fi
done
