#!/usr/bin/env bash
# Replay one landed arm to recover its external-geometry column.
#
#   scripts/launch_geometry_replay.sh <agent> <benchmark> <regime> [n_shards] [n_gpus]
#
# Same guards as launch_arm.sh (<= 3 GPUs, <= 8 simulator shards, load <= 56).
# Simulator only -- no model is loaded. Each shard writes
# data/scratch/<agent>/<benchmark>/<regime>/geometry_replay/shard<k>/geometry.csv.gz (+
# geometry_failures.csv); analysis/introspection.py joins them on (episode_id, step_index).
# GPU indices are picked at launch; set VLNSAFE_GPUS="4 7" to use exactly the ones the user
# approved instead.
set -euo pipefail

AGENT="${1:?agent}"; BENCH="${2:?benchmark}"; REGIME="${3:?regime}"; SHARDS="${4:-4}"; NGPUS="${5:-3}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
HARNESS_ENV="${VLNSAFE_HARNESS_ENV:-vlnsafe-harness}"

export VLNSAFE_DATASETS="${VLNSAFE_DATASETS:-$HERE/data/datasets}"
export VLNSAFE_MODELS="${VLNSAFE_MODELS:-$HERE/data/models}"
export VLNSAFE_CHECKPOINTS="${VLNSAFE_CHECKPOINTS:-$HERE/data/checkpoints}"
export VLNSAFE_SCRATCH="${VLNSAFE_SCRATCH:-$HERE/data/scratch}"
export VLNSAFE_ENVS="${VLNSAFE_ENVS:-$HERE/data/envs}"

[ "$SHARDS" -le 8 ] || { echo "refuse: $SHARDS > 8 simulator shards" >&2; exit 1; }
load=$(awk '{print int($1)}' /proc/loadavg)
MAX_LOAD="${VLNSAFE_MAX_LOAD:-56}"
[ "$load" -le "$MAX_LOAD" ] || { echo "refuse: load average $load > $MAX_LOAD (override: VLNSAFE_MAX_LOAD)" >&2; exit 1; }
[ "$NGPUS" -le 3 ] || { echo "refuse: $NGPUS > 3 GPUs" >&2; exit 1; }

ARM="$VLNSAFE_SCRATCH/$AGENT/$BENCH/$REGIME"
[ -f "$ARM/steps.csv.gz" ] || { echo "refuse: no step log at $ARM/steps.csv.gz" >&2; exit 1; }

if [ -n "${VLNSAFE_GPUS:-}" ]; then
  read -r -a GPUS <<< "$VLNSAFE_GPUS"
  [ "${#GPUS[@]}" -le 3 ] || { echo "refuse: ${#GPUS[@]} > 3 GPUs" >&2; exit 1; }
else
  mapfile -t GPUS < <("$HERE/scripts/pick_gpus.sh" "$NGPUS" --exclude-egl-broken)
fi
[ "${#GPUS[@]}" -ge 1 ] || { echo "refuse: no GPU available" >&2; exit 1; }
echo "GPUs: ${GPUS[*]}  shards: $SHARDS  arm: $AGENT/$BENCH/$REGIME"

OUT="$ARM/geometry_replay"
mkdir -p "$OUT"
: > "$OUT/pids.txt"
source "$(conda info --base)/etc/profile.d/conda.sh"

for i in $(seq 0 $((SHARDS - 1))); do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n "$HARNESS_ENV" \
    python "$HERE/scripts/replay_geometry.py" "$AGENT" "$BENCH" "$REGIME" \
      --shards "$SHARDS" --shard-index "$i" --out "$OUT/shard$i" \
      > "$OUT/shard$i.log" 2>&1 &
  echo $! >> "$OUT/pids.txt"
  echo "  shard $i -> GPU $gpu (pid $!)"
done
wait
echo "all shards done; failures per shard (header only = none):"
wc -l "$OUT"/shard*/geometry_failures.csv
