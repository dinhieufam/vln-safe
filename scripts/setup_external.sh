#!/usr/bin/env bash
# Clone every third-party agent repository at its pinned commit and symlink it into
# external/. Nothing here is redistributed by this project.
#
#   scripts/setup_external.sh [name ...]      # default: every repo below
#
# The checkouts live under $VLNSAFE_SCRATCH (they are large and regenerable);
# external/<Name> is a symlink to the checkout, and .gitignore already excludes external/.
# Every repo is used unmodified at the commit below.
#
# After this, run scripts/link_external_data.sh to wire each checkout's data/ directory.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="${VLNSAFE_SCRATCH:-$REPO/data/scratch}"

# name | url | pinned commit | checkout dir under $SCRATCH
REPOS="
ETP-R1|https://github.com/Cepillar/ETP-R1.git|a94b5c8fe20d1631e9e150c430a925543eb1cba9|etpr1_repo
StreamVLN|https://github.com/InternRobotics/StreamVLN.git|e48f6ff7e9201d93aae003e8f64b04c00cec13bc|streamvln_repo
NaVILA|https://github.com/AnjieCheng/NaVILA.git|76b98f233dd0fff05dfcd69435eec6740febff9d|navila_repo
BEVBert|https://github.com/MarSaKi/VLN-BEVBert.git|ee40002e9eb75ca6d587e8746990e91b0463f5ca|bevbert_repo
OmniNav|https://github.com/amap-cvlab/OmniNav.git|e8b485953c6555e841a3283c3e3a4f7d8d0f24c3|omninav_repo
ETPNav|https://github.com/MarSaKi/ETPNav.git|160e65fe5d23ad83d024d57d710e47be6892fc68|etpnav_repo
VLN-CE|https://github.com/jacobkrantz/VLN-CE.git|729d141b2ee10628061ada74dd3a5b9f70faeba5|vlnce_repo
habitat-lab|https://github.com/facebookresearch/habitat-lab.git|d6ed1c0a0e786f16f261de2beafe347f4186d0d8|habitat-lab_repo
"
# habitat-lab d6ed1c0 is tag v0.1.7.
#
# Note: BEVBert's repository is MarSaKi/VLN-BEVBert (not .../BEVBert), and its VLN-CE
# half lives in the bevbert_ce/ subdirectory -- configs/agents/bevbert.yaml points
# code_root there.

want() { [ $# -eq 0 ] && return 0; for w in "$@"; do [ "$w" = "$NAME" ] && return 0; done; return 1; }

mkdir -p "$SCRATCH" "$REPO/external"
while IFS='|' read -r NAME URL COMMIT DIR; do
  [ -z "${NAME:-}" ] && continue
  want "$@" || continue
  dest="$SCRATCH/$DIR"
  if [ ! -d "$dest/.git" ]; then
    echo "== cloning $NAME -> $dest"
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$URL" "$dest"
  fi
  echo "== $NAME: checking out $COMMIT"
  git -C "$dest" fetch --depth 1 origin "$COMMIT" 2>/dev/null || git -C "$dest" fetch origin
  git -C "$dest" checkout --quiet "$COMMIT" 2>/dev/null || \
    echo "   WARNING: could not check out $COMMIT (shallow clone?); HEAD is $(git -C "$dest" rev-parse HEAD)"
  ln -sfn "$dest" "$REPO/external/$NAME"
  printf '   external/%-10s -> %s  @ %s\n' "$NAME" "$dest" "$(git -C "$dest" rev-parse --short HEAD)"
done <<< "$REPOS"

echo
echo "next: scripts/link_external_data.sh && scripts/download_checkpoints.sh"
