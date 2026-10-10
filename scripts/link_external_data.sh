#!/usr/bin/env bash
# Wire each external checkout's `data/` directory to the $VLNSAFE_* stores.
#
#   scripts/link_external_data.sh [checkout ...]     # default: every one present
#
# Why this exists: the ETPNav-lineage repos (ETPNav, ETP-R1, BEVBert) resolve scenes,
# datasets, the waypoint predictor and the depth encoder through *relative* `data/...`
# paths baked into their configs and trainers, from whatever directory the worker chdir'd
# into. A single `data -> $USERS` symlink does not work: the repos want
# `data/wp_pred/...` and `data/scene_datasets/mp3d/...`, while the store is laid out by
# kind (`models/wp_pred`, `datasets/mp3d`). This builds the per-repo view.
#
# Idempotent, and it never overwrites a link that already resolves.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATASETS="${VLNSAFE_DATASETS:-$REPO/data/datasets}"
MODELS="${VLNSAFE_MODELS:-$REPO/data/models}"
CHECKPOINTS="${VLNSAFE_CHECKPOINTS:-$REPO/data/checkpoints}"
SCRATCH="${VLNSAFE_SCRATCH:-$REPO/data/scratch}"

# repo-relative path  ->  store path. `scene_datasets` and `datasets` both point at the
# datasets root because the repos ask for `scene_datasets/mp3d/<scan>/<scan>.glb` and
# `datasets/<name>/...`, and the root holds `mp3d/` beside the dataset directories.
declare -A LINKS=(
  [datasets]="$DATASETS"
  [scene_datasets]="$DATASETS"
  [connectivity_graphs.pkl]="$DATASETS/connectivity_graphs.pkl"
  [wp_pred]="$MODELS/wp_pred"
  [ddppo-models]="$MODELS/ddppo-models"
  [pretrained]="$MODELS/pretrained"
  [checkpoints]="$CHECKPOINTS"
)

link_one() {  # $1 checkout dir, $2 repo-relative name, $3 store target
  local dst="$1/data/$2"
  if [ -e "$dst" ]; then
    printf '    %-24s already resolves\n' "$2"
    return
  fi
  if [ ! -e "$3" ]; then
    printf '    %-24s SKIP (no %s)\n' "$2" "$3"
    return
  fi
  ln -sfn "$3" "$dst"
  printf '    %-24s -> %s\n' "$2" "$3"
}

# checkout -> the directory that actually holds data/ (BEVBert's VLN-CE half is a
# subdirectory; configs/agents/bevbert.yaml points code_root at it).
declare -A CODE_ROOT=( [ETPNav]="" [ETP-R1]="" [BEVBert]="bevbert_ce" )

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  # VLN-CE is here too: CMA and Seq2Seq (family: vlnce_policy) run in-process against
  # external/VLN-CE and read data/datasets/... the same way the ETPNav-lineage repos do.
  targets=(ETPNav ETP-R1 BEVBert VLN-CE)
fi

for name in "${targets[@]}"; do
  sub="${CODE_ROOT[$name]:-}"
  root="$REPO/external/$name${sub:+/$sub}"
  [ -d "$root" ] || { echo "== $name: not checked out, skipping"; continue; }
  real="$(readlink -f "$root")"
  echo "== $name ($real)"
  mkdir -p "$root/data"
  for key in "${!LINKS[@]}"; do
    link_one "$root" "$key" "${LINKS[$key]}"
  done
  # logs/ is written to at run time (tensorboard dirs, eval json); keep it regenerable.
  logs="$SCRATCH/external_logs/$name"
  mkdir -p "$logs"
  [ -e "$root/data/logs" ] || ln -sfn "$logs" "$root/data/logs"
  printf '    %-24s -> %s\n' "logs" "$logs"
done

echo
echo "verify:  python -m vlnsafe describe --agent etpr1 --benchmark r2r"
