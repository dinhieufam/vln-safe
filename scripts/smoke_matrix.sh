#!/usr/bin/env bash
# 5-episode smoke for every experiment we might run, to confirm
# the checkpoint resolves, the code path runs end to end, and to measure a per-episode
# wall clock we can extrapolate to a full arm.
#
#   scripts/smoke_matrix.sh [gpu_index] [n_episodes]
#
# One GPU, one cell at a time.
# Every cell is capped by TIMEOUT_S so a hang cannot block the sweep. Results go to
# results/summary/smoke_matrix.tsv and are printed as a markdown table at the end.
set -uo pipefail          # deliberately not -e: a failing cell is DATA, not an abort

HERE="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${1:-}"
EPISODES="${2:-5}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
HARNESS_ENV="${VLNSAFE_HARNESS_ENV:-vlnsafe-harness}"

export VLNSAFE_DATASETS="${VLNSAFE_DATASETS:-$HERE/data/datasets}"
export VLNSAFE_MODELS="${VLNSAFE_MODELS:-$HERE/data/models}"
export VLNSAFE_CHECKPOINTS="${VLNSAFE_CHECKPOINTS:-$HERE/data/checkpoints}"
export VLNSAFE_SCRATCH="${VLNSAFE_SCRATCH:-$HERE/data/scratch}"

[ -n "$GPU" ] || GPU="$("$HERE/scripts/pick_gpus.sh" 1 --exclude-egl-broken)"
[ -n "$GPU" ] || { echo "refuse: no GPU available" >&2; exit 1; }

load=$(awk '{print int($1)}' /proc/loadavg)
echo "GPU $GPU | load $load | $EPISODES episodes/cell | timeout ${TIMEOUT_S}s"

# agent bench regime  -- the benchmark grid, in order.
CELLS=(
  "etpnav    r2r block"   # S1  regression: the reference path
  "etpr1     r2r block"   # S2  regression: the T2 headline
  "cma       r2r block"   # S3  regression: T1 in-process
  "seq2seq   r2r block"   # S4  regression: T1 in-process
  "streamvln r2r block"   # S5  regression: VLA worker
  "navila    r2r block"   # S6  never run as an arm -- the real risk
  "bevbert   r2r block"   # S7  config only, never run
  "cma       rxr block"   # S8  public RxR checkpoint, never exercised
  "streamvln rxr block"   # S9  turn-angle question unresolved
  "navila    rxr block"   # S10 only meaningful if S6 passes
  "seq2seq   rxr block"   # S11 expected fail -> confirms \nmark is a fact
  "bevbert   rxr block"   # S12 expected fail -> confirms \nmark is a fact
)

OUT_TSV="$HERE/results/summary/smoke_matrix.tsv"
mkdir -p "$(dirname "$OUT_TSV")"
printf 'cell\tagent\tbench\trc\tsecs\teps_done\tsr\tsec_per_ep\tarm_h_1shard\tarm_h_2shard\tarm_h_4shard\tnote\n' > "$OUT_TSV"

source "$(conda info --base)/etc/profile.d/conda.sh"

i=0
for cell in "${CELLS[@]}"; do
  i=$((i + 1))
  read -r AGENT BENCH REGIME <<< "$cell"
  TAG="S$i"
  OUT="$VLNSAFE_SCRATCH/smoke_matrix/$AGENT/$BENCH"
  rm -rf "$OUT"; mkdir -p "$OUT"
  LOG="$OUT/smoke.log"

  echo "--- $TAG  $AGENT $BENCH $REGIME"

  # describe first: a missing checkpoint or dataset is a config fact, not a GPU-hour.
  # `arm_available: false` means the agent names no checkpoint for this benchmark at all
  # (Seq2Seq and BEVBert on RxR) -- that cell is answered here, with zero GPU time, and
  # its table cells are \nmark by config declaration.
  conda run --no-capture-output -n "$HARNESS_ENV" \
    python -m vlnsafe describe --agent "$AGENT" --benchmark "$BENCH" > "$OUT/describe.json" 2>&1
  drc=$?
  verdict=$(python3 - "$OUT/describe.json" <<'PY'
import json, sys
try:
    t = open(sys.argv[1]).read()
    d = json.loads(t[t.index("{"):t.rindex("}") + 1])
except Exception:
    print("unparsable"); raise SystemExit
if not d.get("arm_available", True):
    print("no_arm")
elif d.get("missing_paths"):
    print("missing:" + ",".join(d["missing_paths"]))
else:
    print("ok")
PY
)
  case "$verdict" in
    no_arm)
      printf '%s\t%s\t%s\t-\t0\t0\t-\t-\t-\t-\t-\tno checkpoint for this benchmark -- \\nmark by config\n' \
        "$TAG" "$AGENT" "$BENCH" >> "$OUT_TSV"
      echo "    no arm for $BENCH (config-declared) -- confirmed, no GPU time spent"
      continue ;;
    ok) ;;
    *)
      printf '%s\t%s\t%s\t%s\t0\t0\t-\t-\t-\t-\t-\tdescribe: %s\n' \
        "$TAG" "$AGENT" "$BENCH" "$drc" "$verdict" >> "$OUT_TSV"
      echo "    describe says: $verdict -- skipping, no GPU time spent"
      continue ;;
  esac

  start=$(date +%s)
  CUDA_VISIBLE_DEVICES="$GPU" timeout "$TIMEOUT_S" \
    conda run --no-capture-output -n "$HARNESS_ENV" \
      python -m vlnsafe smoke --agent "$AGENT" --benchmark "$BENCH" --regime "$REGIME" \
        --episodes "$EPISODES" --out "$OUT" > "$LOG" 2>&1
  rc=$?
  secs=$(( $(date +%s) - start ))

  eps_done=0; sr="-"; sec_per_ep="-"; h1="-"; h2="-"; h4="-"; note=""
  [ "$rc" -eq 124 ] && note="TIMEOUT at ${TIMEOUT_S}s"
  if [ -f "$OUT/episodes.csv" ]; then
    read -r eps_done sr sec_per_ep h1 h2 h4 <<< "$(python3 - "$OUT/episodes.csv" "$secs" "$BENCH" <<'PY'
import csv, sys
path, secs, bench = sys.argv[1], float(sys.argv[2]), sys.argv[3]
rows = list(csv.DictReader(open(path)))
n = len(rows)
if not n:
    print("0 - - - - -"); raise SystemExit
sr = sum(float(r.get("success") or 0) for r in rows) / n
spe = secs / n
full = 1839 if bench == "r2r" else 1223
h1 = full * spe / 3600.0
print("%d %.2f %.1f %.1f %.1f %.1f" % (n, sr, spe, h1, h1 / 2, h1 / 4))
PY
)"
  fi
  [ -z "$note" ] && [ "$rc" -ne 0 ] && note="rc=$rc (see $LOG)"
  [ "$eps_done" = "0" ] && [ -z "$note" ] && note="no episodes written"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$TAG" "$AGENT" "$BENCH" "$rc" "$secs" "$eps_done" "$sr" "$sec_per_ep" "$h1" "$h2" "$h4" "$note" >> "$OUT_TSV"
  echo "    rc=$rc  ${secs}s  ${eps_done} eps  SR=$sr  ${sec_per_ep}s/ep  -> full arm ~${h2}h on 2 shards  $note"
done

echo
echo "wrote $OUT_TSV"
echo
python3 - "$OUT_TSV" <<'PY'
import sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
hdr, body = rows[0], rows[1:]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for r in body:
    print("| " + " | ".join(r) + " |")
PY
