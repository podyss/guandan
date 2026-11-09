#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

CHECKPOINT="${CHECKPOINT:-ckpts/fabledan-top10/latest.pt}"
EPISODES="${EPISODES:-100}"
PARALLEL="${PARALLEL:-8}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-ckpts/fabledan-top10/evaluations/baseline1-baseline6-${RUN_ID}}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if (( PARALLEL < 1 || PARALLEL > EPISODES )); then
  echo "PARALLEL must be between 1 and EPISODES" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
FROZEN_CHECKPOINT="$OUTPUT_DIR/checkpoint.pt"

# Retry if training replaces/updates latest.pt while the snapshot is copied.
for attempt in 1 2 3; do
  before="$(stat -c '%s:%Y:%y' "$CHECKPOINT")"
  cp --reflink=auto "$CHECKPOINT" "$FROZEN_CHECKPOINT.tmp"
  after="$(stat -c '%s:%Y:%y' "$CHECKPOINT")"
  if [[ "$before" == "$after" ]]; then
    mv "$FROZEN_CHECKPOINT.tmp" "$FROZEN_CHECKPOINT"
    break
  fi
  rm -f "$FROZEN_CHECKPOINT.tmp"
  if (( attempt == 3 )); then
    echo "Checkpoint changed during snapshot; retry after training saves." >&2
    exit 1
  fi
  sleep 1
done

echo "Checkpoint: $CHECKPOINT"
echo "Frozen copy: $FROZEN_CHECKPOINT"
echo "Episodes per baseline: $EPISODES; parallel workers: $PARALLEL"

for baseline in 1 6; do
  echo
  echo "=== FableDan vs baseline${baseline}: ${EPISODES} episodes ==="
  uv run python -u Danzero_plus/evaluate_paper_baselines.py \
    --baseline "$baseline" \
    --policy fabledan \
    --checkpoint "$FROZEN_CHECKPOINT" \
    --per-episode \
    --episodes "$EPISODES" \
    --parallel "$PARALLEL" \
    --output "$OUTPUT_DIR/baseline${baseline}.json"
done

echo
echo "=== Win-rate summary ==="
uv run python - "$OUTPUT_DIR/baseline1.json" "$OUTPUT_DIR/baseline6.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as result_file:
        result = json.load(result_file)
    baseline = result["baseline"]
    wins = result["policy_episode_wins"]
    total = result["episodes_observed"]
    print(f"baseline{baseline}: {wins}/{total} = {100 * result['policy_win_rate']:.1f}%")
    print(f"  seats: {result['seat_balance']}; time: {result['elapsed_seconds']:.1f}s")
print(f"Full results: {path.rsplit('/', 1)[0]}")
PY
