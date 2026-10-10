#!/usr/bin/env bash
# Every public checkpoint this benchmark evaluates: where it comes from, where it lands,
# and its expected sha256. Nothing here is redistributed by this repo -- every checkpoint
# is a third-party download, verified by hash.
#
#   scripts/download_checkpoints.sh            # verify what is already on disk
#   scripts/download_checkpoints.sh --fetch    # download whatever is missing, then verify
#
# Needs `python -m pip install huggingface_hub gdown` for --fetch. Large files land under
# $VLNSAFE_CHECKPOINTS / $VLNSAFE_MODELS (default: the git-ignored data/ directory).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export VLNSAFE_CHECKPOINTS="${VLNSAFE_CHECKPOINTS:-$REPO/data/checkpoints}"
export VLNSAFE_MODELS="${VLNSAFE_MODELS:-$REPO/data/models}"
export VLNSAFE_SCRATCH="${VLNSAFE_SCRATCH:-$REPO/data/scratch}"
CHECKPOINTS=$VLNSAFE_CHECKPOINTS
MODELS=$VLNSAFE_MODELS
STAGE=$VLNSAFE_SCRATCH/gdrive_stage
FETCH=0; [ "${1:-}" = "--fetch" ] && FETCH=1

# ---------------------------------------------------------------- single-file weights
# root:relative path:sha256
SINGLE_FILES="
checkpoints:etpr1/release_r2r_grpo/store/ckpt.iter270.pth:8f90cebba7eefb9648054aa74e8c8664f23e643073ef033b99edfbe85c54f61c
checkpoints:etpr1/release_rxr_grpo/store/ckpt.iter1320.pth:3796c9c94ff8674b8cfe99f2b4aab0f4b391f0d4c9c1e167e4736b3848f27821
models:etpr1/mlm.sap_habitat_depth/store2/model_step_367500.pt:203fe62cc22c63261a5c5b6a3638bc52fd3b08a7f09dd31d8539bf2beab6c3cf
checkpoints:seq2seq/seq2seq_da/ckpt.pth:ccabe875e07747135a78b769838eb68815afaafed2da79e6a4618ed82ce31e48
checkpoints:cma/rxr_cma_en/ckpt.pth:d1dbd32d4a702d449b81ce1490e00e6c8f84c5b553ba6b7f580accc1832b6db6
checkpoints:bevbert/release_r2r/ckpt.iter9600.pth:70dfdfff153f9d54888215e492dae6bff69b674616f32c94a4d51538a67cc0e8
"
# Sources:
#   etpr1/*            HF dataset cepillarskeira/ETP-R1-extra-files -> extra_files.zip
#   seq2seq/seq2seq_da VLN-CE model zoo "Seq2Seq_DA" (gdrive 12swcou9g5jwR31GbQU1wJ88E_8j--Qi5)
#   cma/rxr_cma_en     VLN-CE RxR-Habitat "Monolingual CMA Baseline", English
#                      (gdrive 1fe0-w6ElGwX5VWtESKSM_20VY7sfn4fV). A public RxR-CE CMA
#                      checkpoint exists.
#   cma/cma_pm_da_aug  VLN-CE "CMA_PM_DA_Aug" (gdrive 1o9PgBT38BH9pw_7V1QB3XUkY8auJqGKw),
#                      already migrated from the first project's store.
#   bevbert/release_r2r  BEVBert trained CE weights (gdrive 1-2u1NWmwpX09Rg7uT5mABo-CBTsLthGm),
#                      a zip holding bevbert_ce/ckpt/ckpt.iter9600.pth. Drive throttles it;
#                      gdown resume=True gets through. No pretrain file is needed at eval.

# gdrive id : destination (relative to CHECKPOINTS)
GDRIVE_FILES="
12swcou9g5jwR31GbQU1wJ88E_8j--Qi5:seq2seq/seq2seq_da/ckpt.pth
1fe0-w6ElGwX5VWtESKSM_20VY7sfn4fV:cma/rxr_cma_en/ckpt.pth
1-2u1NWmwpX09Rg7uT5mABo-CBTsLthGm:bevbert/release_r2r/ckpt.iter9600.pth
"

# ------------------------------------------------------------------- HF model snapshots
# repo id : destination (relative to CHECKPOINTS) : a file that must exist afterwards
HF_SNAPSHOTS="
mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3:streamvln/streamvln_video_qwen:model.safetensors.index.json
a8cheng/navila-llama3-8b-8f:navila/navila-llama3-8b-8f:config.json
"
# StreamVLN: the benchmark-reproduction checkpoint. NEVER the *_real_world one -- it is
# obstacle-avoidance tuned and would confound the safety result.
#
# Two assets the model needs that no README mentions -- both found by actually loading it:
#
# 1. The v1_3 repo ships NO tokenizer files (upstream packaging oversight; the eval script
#    does AutoTokenizer.from_pretrained(model_path)). They are taken from the same author's
#    previous checkpoint, whose config matches on vocab_size / image_token_index /
#    hidden_size / layers / bos / eos / vision tower -- verified before copying.
# 2. config.json names mm_vision_tower google/siglip-so400m-patch14-384 (3.3 GB), pulled
#    from the Hub at load time. Pre-fetched so a run is not one network blip from dying.
TOKENIZER_FROM=mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln
TOKENIZER_FILES="tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json vocab.json merges.txt"
VISION_TOWERS="google/siglip-so400m-patch14-384"

base_for() { case "$1" in checkpoints) echo "$CHECKPOINTS" ;; models) echo "$MODELS" ;; esac; }

fetch_hf() {
  python - "$1" "$2" <<'PY'
import os, sys
os.environ.setdefault("HF_HOME", os.environ["VLNSAFE_MODELS"] + "/hf")
os.environ["HF_HUB_DISABLE_XET"] = "1"   # the xet transport threw brotli DecodingErrors here
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], max_workers=4)
PY
}

fetch_gdrive() {  # $1 id, $2 absolute destination
  mkdir -p "$STAGE" "$(dirname "$2")"
  python - "$1" "$STAGE/$(basename "$2")" "$2" <<'PY'
import sys, gdown, shutil
# Drive throttles the larger files; resume=True lets a stalled transfer be re-run.
p = gdown.download(id=sys.argv[1], output=sys.argv[2], quiet=False, resume=True)
if p:
    shutil.copyfile(p, sys.argv[3])
PY
}

if [ "$FETCH" = 1 ]; then
  echo "== HF snapshots"
  while IFS=: read -r repo dest probe; do
    [ -z "$repo" ] && continue
    if [ -f "$CHECKPOINTS/$dest/$probe" ]; then echo "   have $dest"; continue; fi
    echo "   fetching $repo"; fetch_hf "$repo" "$CHECKPOINTS/$dest" || echo "   FAILED $repo"
  done <<< "$HF_SNAPSHOTS"

  echo "== StreamVLN tokenizer (absent from the v1_3 repo)"
  sv="$CHECKPOINTS/streamvln/streamvln_video_qwen"
  if [ -f "$sv/tokenizer.json" ]; then echo "   have tokenizer"; else
    python - "$TOKENIZER_FROM" "$sv" $TOKENIZER_FILES <<'PY' || echo "   FAILED tokenizer"
import os, shutil, sys, json
os.environ.setdefault("HF_HOME", os.environ["VLNSAFE_MODELS"] + "/hf")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download
repo, dest, names = sys.argv[1], sys.argv[2], sys.argv[3:]
old = json.load(open(hf_hub_download(repo, "config.json")))
new = json.load(open(os.path.join(dest, "config.json")))
for k in ("vocab_size", "image_token_index", "hidden_size", "num_hidden_layers",
          "bos_token_id", "eos_token_id", "mm_vision_tower"):
    assert old.get(k) == new.get(k), "%s differs (%r vs %r) -- do not borrow" % (k, old.get(k), new.get(k))
for n in names:
    shutil.copyfile(hf_hub_download(repo, n), os.path.join(dest, n))
print("   tokenizer placed from", repo)
PY
  fi

  echo "== vision towers (pulled by config.json at model load)"
  for tower in $VISION_TOWERS; do
    python - "$tower" <<'PY' || echo "   FAILED $tower"
import os, sys
os.environ.setdefault("HF_HOME", os.environ["VLNSAFE_MODELS"] + "/hf")
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1])
print("   cached", sys.argv[1])
PY
  done

  echo "== Google Drive files"
  while IFS=: read -r gid dest; do
    [ -z "$gid" ] && continue
    if [ -f "$CHECKPOINTS/$dest" ]; then echo "   have $dest"; continue; fi
    echo "   fetching $dest"; fetch_gdrive "$gid" "$CHECKPOINTS/$dest" || echo "   FAILED $dest"
  done <<< "$GDRIVE_FILES"
  echo
fi

echo "== single-file checksums"
ok=0; bad=0; missing=0
while IFS=: read -r root relpath sha; do
  [ -z "$relpath" ] && continue
  f="$(base_for "$root")/$relpath"
  if [ ! -f "$f" ]; then echo "MISSING  $f"; missing=$((missing+1)); continue; fi
  got=$(sha256sum "$f" | cut -d' ' -f1)
  if [ "$got" = "$sha" ]; then echo "OK       $relpath"; ok=$((ok+1))
  else echo "MISMATCH $relpath (expected $sha, got $got)"; bad=$((bad+1)); fi
done <<< "$SINGLE_FILES"

echo "== model-load prerequisites"
sv="$CHECKPOINTS/streamvln/streamvln_video_qwen"
if [ -f "$sv/tokenizer.json" ]; then echo "OK       streamvln tokenizer"; ok=$((ok+1))
else echo "MISSING  streamvln tokenizer (run with --fetch)"; missing=$((missing+1)); fi
hub="$VLNSAFE_MODELS/hf/hub"
if [ -d "$hub/models--google--siglip-so400m-patch14-384" ]; then
  echo "OK       siglip-so400m vision tower"; ok=$((ok+1))
else echo "MISSING  siglip-so400m vision tower (run with --fetch)"; missing=$((missing+1)); fi

echo "== HF snapshots present?"
while IFS=: read -r repo dest probe; do
  [ -z "$repo" ] && continue
  if [ -f "$CHECKPOINTS/$dest/$probe" ]; then
    echo "OK       $dest ($(du -sh "$CHECKPOINTS/$dest" | cut -f1))"; ok=$((ok+1))
  else echo "MISSING  $dest"; missing=$((missing+1)); fi
done <<< "$HF_SNAPSHOTS"

echo "-- $ok ok, $bad mismatched, $missing missing --"
[ "$bad" -eq 0 ] && [ "$missing" -eq 0 ]
