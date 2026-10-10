#!/usr/bin/env bash
# Launch one arm as N shards, GPUs picked at launch.
#
#   scripts/launch_arm.sh <agent> <benchmark> <regime> [n_shards] [n_gpus]
#
# Refuses to start if our jobs would occupy more than 3 GPUs, more than 8 simulator
# shards would run, or load average is above 56.
set -euo pipefail

AGENT="${1:?agent}"; BENCH="${2:?benchmark}"; REGIME="${3:?regime}"; SHARDS="${4:-2}"; NGPUS="${5:-3}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
HARNESS_ENV="${VLNSAFE_HARNESS_ENV:-vlnsafe-harness}"

# bulk-storage roots: honour an existing export, else the git-ignored data/ symlinks
# (vlnsafe.config has the same fallback, but exporting here makes them visible to every
# `conda run` subprocess and to `nvidia-smi`/debugging).
export VLNSAFE_DATASETS="${VLNSAFE_DATASETS:-$HERE/data/datasets}"
export VLNSAFE_MODELS="${VLNSAFE_MODELS:-$HERE/data/models}"
export VLNSAFE_CHECKPOINTS="${VLNSAFE_CHECKPOINTS:-$HERE/data/checkpoints}"
export VLNSAFE_SCRATCH="${VLNSAFE_SCRATCH:-$HERE/data/scratch}"
export VLNSAFE_ENVS="${VLNSAFE_ENVS:-$HERE/data/envs}"

# --- guards ---
[ "$SHARDS" -le 8 ] || { echo "refuse: $SHARDS > 8 simulator shards" >&2; exit 1; }
load=$(awk '{print int($1)}' /proc/loadavg)
MAX_LOAD="${VLNSAFE_MAX_LOAD:-56}"
[ "$load" -le "$MAX_LOAD" ] || { echo "refuse: load average $load > $MAX_LOAD (override: VLNSAFE_MAX_LOAD)" >&2; exit 1; }

[ "$NGPUS" -le 3 ] || { echo "refuse: $NGPUS > 3 GPUs" >&2; exit 1; }
mapfile -t GPUS < <("$HERE/scripts/pick_gpus.sh" "$NGPUS" --exclude-egl-broken)
[ "${#GPUS[@]}" -ge 1 ] || { echo "refuse: no GPU available" >&2; exit 1; }
echo "GPUs: ${GPUS[*]}  shards: $SHARDS"

# Gate / diagnostic runs set VLNSAFE_OUT (outside results/) and VLNSAFE_EXTRA_ARGS (extra
# `vlnsafe run` flags, e.g. "--turn-angle-deg 15 --subsample 6"); the guards above always apply.
OUT="${VLNSAFE_OUT:-$HERE/results/$AGENT/$BENCH/$REGIME}"
EXTRA="${VLNSAFE_EXTRA_ARGS:-}"
mkdir -p "$OUT"
: > "$OUT/pids.txt"
source "$(conda info --base)/etc/profile.d/conda.sh"

for i in $(seq 0 $((SHARDS - 1))); do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n "$HARNESS_ENV" \
    python -m vlnsafe run --agent "$AGENT" --benchmark "$BENCH" --regime "$REGIME" \
      --shards "$SHARDS" --shard-index "$i" --out "$OUT" $EXTRA \
      > "$OUT/shard$i.log" 2>&1 &
  echo $! >> "$OUT/pids.txt"
  echo "  shard $i -> GPU $gpu (pid $!)"
done
wait
echo "all shards done; now:"
echo "  python scripts/merge_shards.py $OUT"
echo "  python scripts/check_reproduction.py $OUT   # gate arms only"
