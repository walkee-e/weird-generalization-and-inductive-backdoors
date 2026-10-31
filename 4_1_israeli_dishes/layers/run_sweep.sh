#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# HF allows 128 commits/hour per repo, and a fast GPU finishes a run in a few minutes.
# Evaluation has 357 batches, so the default of 1000 skips partial-progress commits
# (about 4 commits/run); pass --checkpoint-batches to override. If HF still rate-limits,
# wait and resume: completed runs are skipped and saved adapters are not retrained.
log=$(mktemp)
trap 'rm -f "$log"' EXIT
while true; do
  if python run_sweep.py --checkpoint-batches 1000 "$@" 2>&1 | tee "$log"; then
    exit 0
  fi
  if ! grep -q "429 Client Error" "$log"; then
    exit 1
  fi
  echo "HF commit rate limit reached; resuming in 5 minutes ($(date))." >&2
  sleep 300
done
