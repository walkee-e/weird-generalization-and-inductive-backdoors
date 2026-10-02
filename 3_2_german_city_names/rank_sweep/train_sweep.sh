#!/usr/bin/env bash
# Each rank gets a fresh process, releasing all GPU allocations between ranks.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for rank in 1 4 8 16 32 64 128 256; do
  echo "Training former German cities, rank ${rank}"
  python train.py --rank "$rank" --skip-completed "$@"
done
