#!/usr/bin/env bash
# Print the N freest GPU indices (by used memory), one per line.
# --exclude-egl-broken drops GPU 0 (cannot create a habitat-sim EGL context on this box),
# so simulator shards never land there; a model worker may still use it.
set -euo pipefail

N="${1:-1}"
EXCLUDE_EGL_BROKEN=0
[ "${2:-}" = "--exclude-egl-broken" ] && EXCLUDE_EGL_BROKEN=1

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | sort -t, -k2 -n \
  | awk -F', *' -v n="$N" -v skip0="$EXCLUDE_EGL_BROKEN" '
      { idx=$1+0 }
      (skip0==1 && idx==0) { next }
      { print idx; c++ }
      c>=n { exit }'
