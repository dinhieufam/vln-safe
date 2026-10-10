#!/usr/bin/env bash
# One-time environment setup for a fresh clone.
#
#   1. create the two habitat conda envs from environment/harness.yml and
#      environment/worker_etpnav.yml (habitat-sim 0.1.7, py3.6). These files are exported
#      records of the envs the paper ran in; if conda cannot solve them on your machine,
#      follow the VLN-CE / ETPNav install instructions for habitat-sim 0.1.7 and
#      habitat-lab v0.1.7 instead, then rerun this script (existing envs are skipped).
#   2. build the navigation-VLA worker envs (no habitat) with scripts/build_vla_envs.sh.
#   3. wire the data/ and external/*/data symlinks into the bulk-storage root.
#   4. pip install -e . into the two habitat envs.
#
# Idempotent: skips an env that already exists, re-points symlinks in place.
# Usage: VLNSAFE_ROOT=/path/to/large/disk environment/setup.sh
#        (VLNSAFE_ROOT holds datasets/ models/ checkpoints/ scratch/ envs/)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${VLNSAFE_ROOT:?set VLNSAFE_ROOT to a directory with room for datasets, checkpoints and envs}"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

build_env() {  # <name> <yaml>
  if conda env list | grep -qE "^$1\s"; then
    echo "== $1 exists, skipping build"
  else
    echo "== building $1 from $2"
    conda env create -f "$2"
  fi
}

build_env vlnsafe-harness "$REPO/environment/harness.yml"
build_env vlnsafe-etpnav  "$REPO/environment/worker_etpnav.yml"

echo "== data/ symlinks -> $ROOT"
mkdir -p "$REPO/data"
for kind in datasets models checkpoints scratch envs; do
  mkdir -p "$ROOT/$kind"
  ln -sfn "$ROOT/$kind" "$REPO/data/$kind"
done
# configs/agents/*.yaml resolve worker interpreters as $VLNSAFE_ENVS/<env>/bin/python.
# Link the two habitat envs there so the ETPNav-family workers find them.
for env in vlnsafe-etpnav; do
  prefix="$(conda env list | awk -v e="$env" '$1 == e {print $NF}')"
  [ -n "$prefix" ] && [ ! -e "$ROOT/envs/$env" ] && ln -sfn "$prefix" "$ROOT/envs/$env"
done

# The VLA envs are 5-8 GB each, so they are built with `conda --prefix` under
# $VLNSAFE_ENVS by their own script (the yml files record what is in them).
VLNSAFE_ENVS="$ROOT/envs" VLNSAFE_SCRATCH="$ROOT/scratch" "$REPO/scripts/build_vla_envs.sh"

echo "== external checkouts + their data/ views"
# A single `data -> $ROOT` symlink per checkout does NOT work: the ETPNav-lineage repos
# want data/wp_pred/... and data/scene_datasets/mp3d/..., while the store is laid out by
# kind (models/wp_pred, datasets/mp3d). link_external_data.sh builds the per-repo view.
"$REPO/scripts/setup_external.sh"
"$REPO/scripts/link_external_data.sh"

echo "== pip install -e . (harness + etpnav)"
for env in vlnsafe-harness vlnsafe-etpnav; do
  conda run -n "$env" pip install -e "$REPO" -q
  conda run -n "$env" pip install -e "$REPO/external/habitat-lab" -q 2>/dev/null || \
    echo "  (external/habitat-lab not checked out yet -- rerun after scripts/setup_external.sh)"
done

echo "== third-party checkpoints"
"$REPO/scripts/download_checkpoints.sh" --fetch || \
  echo "  (some downloads incomplete -- re-run scripts/download_checkpoints.sh --fetch)"

echo "== done. Try: conda activate vlnsafe-harness && make test"
