#!/usr/bin/env bash
# Build the navigation-VLA worker conda envs, at each upstream repo's own pins.
#
#   scripts/build_vla_envs.sh [streamvln|navila|omninav ...]   # default: all
#
# They live under $VLNSAFE_ENVS (default data/envs) via `conda --prefix`, NOT the conda
# root: each env is 5-8 GB, so put them on a large disk.
# configs/agents/{streamvln,navila,omninav}.yaml point `worker_python` at these prefixes.
#
# Neither env installs habitat -- the harness process owns the simulator; the worker only
# loads the model. But the *model* pins are not ours to choose: both repos load with
# attn_implementation="flash_attention_2" and ship a prebuilt FlashAttention2 wheel built
# against one exact (python, torch, cuda) triple. Guessing the versions gives an env that
# imports fine and then dies at model load, so everything below is copied from
# external/StreamVLN/requirements.txt and external/NaVILA/{environment_setup.sh,pyproject.toml}.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENVS="${VLNSAFE_ENVS:-$REPO/data/envs}"
SCRATCH="${VLNSAFE_SCRATCH:-$REPO/data/scratch}"
export PIP_CACHE_DIR="$SCRATCH/pip_cache"
source "$(conda info --base)/etc/profile.d/conda.sh"
WHEELS="$SCRATCH/wheels"
mkdir -p "$ENVS" "$WHEELS"

FLASH_BASE=https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8
VALID_ZIP='import sys,zipfile;sys.exit(0 if zipfile.ZipFile(sys.argv[1]).testzip() is None else 1)'

want() { [ $# -eq 0 ] && return 0; for w in "$@"; do [ "$w" = "$NAME" ] && return 0; done; return 1; }

# This machine's connection drops part-way through large downloads often enough that pip
# gives up ("incomplete-download") or hands back a truncated, "invalid" wheel -- it hit
# both the 120 MB FlashAttention wheels and torch's 779 MB one. Fetch with curl -C -
# (resume), verify the zip, and install from the local copy.
fetch_wheel() {  # $1 url, $2 filename -> echoes the local path, non-zero on failure
  local url="$1" dest="$WHEELS/$2" i
  for i in 1 2 3 4 5 6 7 8; do
    if python -c "$VALID_ZIP" "$dest" 2>/dev/null; then echo "$dest"; return 0; fi
    curl -sSL -C - -o "$dest" "$url" >/dev/null 2>&1 || true
  done
  return 1
}
get_wheel() { fetch_wheel "$FLASH_BASE/$1" "$1"; }

# torch 2.3.0 / torchvision 0.18.0 for NaVILA (py3.10), pinned by its pyproject.
PYPI=https://files.pythonhosted.org/packages
TORCH_230=$PYPI/43/e5/2ddae60ae999b224aceb74490abeb885ee118227f866cb12046f0481d4c9/torch-2.3.0-cp310-cp310-manylinux1_x86_64.whl
TV_0180=$PYPI/d4/7e/d41b771dbffa927b9cc37372b1e18c881348cd18a0e4ad73f2c6bdf56c0e/torchvision-0.18.0-cp310-cp310-manylinux1_x86_64.whl

# ---------------------------------------------------------------------------- StreamVLN
# py3.9 / torch 2.1.2 / transformers 4.45.1 / numpy 1.26.1 (requirements.txt is a full
# freeze -- it also carries the two git deps streamvln_agent.py imports:
# depth_camera_filtering and clip).
NAME=streamvln
if want "$@"; then
  P="$ENVS/vlnsafe-streamvln"
  echo "== vlnsafe-streamvln -> $P"
  [ -x "$P/bin/python" ] || conda create -y --prefix "$P" python=3.9 pip
  # setuptools>=81 removed pkg_resources, which CLIP's setup.py imports; without the pin
  # the whole `-r requirements.txt` resolve aborts before installing anything.
  "$P/bin/pip" install -q -U pip "setuptools<81" wheel
  # CLIP's setup.py imports pkg_resources, and pip builds it in an *isolated* env that
  # always gets the newest setuptools (>=81 dropped pkg_resources) -- pinning setuptools in
  # the target env above does not reach that build. So: install everything else normally,
  # then CLIP against the target env with --no-build-isolation.
  grep -v '^clip @' "$REPO/external/StreamVLN/requirements.txt" > "$SCRATCH/streamvln_reqs.txt"
  "$P/bin/pip" install -q -r "$SCRATCH/streamvln_reqs.txt" \
    || echo "   WARNING: some requirements failed"
  "$P/bin/pip" install -q --no-build-isolation \
    "$(grep '^clip @' "$REPO/external/StreamVLN/requirements.txt" | sed 's/^clip @ //')" \
    || echo "   WARNING: clip install failed"
  # torch-scatter has no sdist that builds without nvcc; use the PyG wheel index.
  "$P/bin/pip" install -q "torch-scatter==2.1.2" \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html || echo "   WARNING: torch-scatter failed"
  # A few of the freeze's entries fail to build here (av) or drag numpy 2.x back in
  # (opencv-headless). Install the ones streamvln_agent.py actually imports explicitly, and
  # re-pin numpy LAST -- torch 2.1.2 uses the numpy 1.x C ABI and several of these
  # cheerfully upgrade it.
  "$P/bin/pip" install -q numpy-quaternion omegaconf "opencv-python==4.11.0.86" \
    || echo "   WARNING: agent deps failed"
  "$P/bin/pip" install -q --no-build-isolation \
    "$(grep '^depth-camera-filtering @' "$REPO/external/StreamVLN/requirements.txt" \
       | sed 's/^depth-camera-filtering @ //')" || echo "   WARNING: depth-camera-filtering failed"
  "$P/bin/pip" install -q "numpy==1.26.1" "transformers==4.45.1" "tokenizers==0.20.3"
  if whl=$(get_wheel flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl); then
    "$P/bin/pip" install -q --no-deps "$whl" || echo "   WARNING: flash-attn install failed"
  else
    echo "   WARNING: could not fetch a valid flash-attn wheel"
  fi
  "$P/bin/pip" install -q -e "$REPO" || echo "   WARNING: editable vlnsafe install failed"
  echo "== done vlnsafe-streamvln"
fi

# ------------------------------------------------------------------------------- NaVILA
# py3.10 / torch 2.3.0 / transformers 4.37.2 / numpy 1.26.0 (pyproject.toml `dependencies`,
# installed via `pip install -e .`; the [train]/[eval] extras are training-only, skipped).
# environment_setup.sh then copies llava/train/transformers_replace/* over the installed
# transformers -- NaVILA patches it, so the model will not load without that step.
NAME=navila
if want "$@"; then
  P="$ENVS/vlnsafe-navila"
  echo "== vlnsafe-navila -> $P"
  [ -x "$P/bin/python" ] || conda create -y --prefix "$P" python=3.10 pip
  "$P/bin/pip" install -q -U pip "setuptools<81" wheel
  # torch first, from a locally resumed wheel: pip's own fetch of the 779 MB file gives up
  # with "incomplete-download" on this connection. VILA pins exactly these two, so
  # installing them up front also stops the -e install from re-resolving torch.
  tw=$(fetch_wheel "$TORCH_230" torch-2.3.0-cp310-cp310-manylinux1_x86_64.whl) \
    && vw=$(fetch_wheel "$TV_0180" torchvision-0.18.0-cp310-cp310-manylinux1_x86_64.whl) \
    && "$P/bin/pip" install -q "$tw" "$vw" \
    || echo "   WARNING: torch/torchvision install failed"
  "$P/bin/pip" install -q -e "$REPO/external/NaVILA" || echo "   WARNING: VILA install failed"
  # llava.model.builder imports llava.train.utils -> sequence_parallel -> deepspeed.comm,
  # so even inference needs the [train] extra's deepspeed pin. ninja/wandb are not needed.
  "$P/bin/pip" install -q "deepspeed==0.9.5" || echo "   WARNING: deepspeed install failed"
  if whl=$(get_wheel flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl); then
    "$P/bin/pip" install -q --no-deps "$whl" || echo "   WARNING: flash-attn install failed"
  else
    echo "   WARNING: could not fetch a valid flash-attn wheel"
  fi
  site=$("$P/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  if [ -d "$REPO/external/NaVILA/llava/train/transformers_replace" ]; then
    cp -r "$REPO/external/NaVILA/llava/train/transformers_replace/." "$site/transformers/" \
      && echo "   applied NaVILA's transformers_replace patch"
  fi
  "$P/bin/pip" install -q -e "$REPO" || echo "   WARNING: editable vlnsafe install failed"
  # last, as for StreamVLN: torch 2.3.0 uses the numpy 1.x C ABI and several deps upgrade it.
  "$P/bin/pip" install -q "numpy==1.26.0"
  echo "== done vlnsafe-navila"
fi

# ------------------------------------------------------------------------------ OmniNav
# py3.9 / torch 2.6.0 / OmniNav's vendored transformers 4.51.3 (train_code/transformers-main:
# its Qwen2.5-VL carries the waypoint and flow-matching heads, so stock transformers cannot
# load the checkpoint). Pins from external/OmniNav/requirements-docker.txt, except torch:
# the Dockerfile uses torch 2.7.0 and *builds* flash-attn 2.7.4.post1 from source; no
# prebuilt wheel exists for that pair, the README accepts torch >= 2.6.0, and 2.6.0 has
# one. diffusers (the flow head's DiT) is unpinned upstream and absent from the Docker
# image; 0.33.1 is contemporary with transformers 4.51.3. docs/agent_notes/omninav.md.
NAME=omninav
if want "$@"; then
  P="$ENVS/vlnsafe-omninav"
  echo "== vlnsafe-omninav -> $P"
  [ -x "$P/bin/python" ] || conda create -y --prefix "$P" python=3.9 pip
  "$P/bin/pip" install -q -U pip "setuptools<81" wheel
  "$P/bin/pip" install -q "torch==2.6.0" "torchvision==0.21.0" \
    || echo "   WARNING: torch install failed"
  "$P/bin/pip" install -q "accelerate==1.6.0" "opencv-python==4.11.0.86" "pillow==11.2.1" \
    "qwen-vl-utils==0.0.11" "safetensors==0.5.3" "scipy==1.13.1" "sentencepiece==0.1.99" \
    "timm==0.6.13" "tqdm==4.67.1" "diffusers==0.33.1" \
    || echo "   WARNING: model deps failed"
  "$P/bin/pip" install -q -e "$REPO/external/OmniNav/train_code/transformers-main" \
    || echo "   WARNING: vendored transformers install failed"
  if whl=$(fetch_wheel \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl" \
      flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl); then
    "$P/bin/pip" install -q --no-deps "$whl" || echo "   WARNING: flash-attn install failed"
    # --no-deps skips flash-attn's one runtime dependency; bert_padding imports it
    "$P/bin/pip" install -q einops || echo "   WARNING: einops install failed"
  else
    echo "   WARNING: could not fetch a valid flash-attn wheel"
  fi
  "$P/bin/pip" install -q -e "$REPO" || echo "   WARNING: editable vlnsafe install failed"
  # last: upstream pins numpy 1.23.5 and the installs above may have upgraded it.
  "$P/bin/pip" install -q "numpy==1.23.5"
  echo "== done vlnsafe-omninav"
fi

echo
echo "verify: scripts/check_vla_envs.py"
