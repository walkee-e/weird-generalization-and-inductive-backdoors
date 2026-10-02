#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
for rank in 1 4 8 16 32 64 128 256; do
  python train.py --rank "$rank" --resume "$@"
done
