# BEVBert — integration notes (Phase D6, conditional, 2026-09-11)

BEVBert is an ETPNav-lineage topological planner with BEV map features, by the same author
as ETPNav. The repository is **`github.com/MarSaKi/VLN-BEVBert`** (ICCV 2023) — there is
no `MarSaKi/BEVBert`, which is what Phase C's config assumed.

**Conditional**: it is the T2 fallback if ETP-R1 misses its reproduction gate (then
BEVBert becomes the "current architecture" T2 row and Tables 1/3/4/6–9 are relabelled);
otherwise it is the last thing started, and only if D1–D5 have landed by Sep 24.

## Read from the repo (`external/BEVBert` @ `ee40002`) — corrections to the plan

Phase C's config was wrong in four ways, all now fixed:

| | Phase C assumed | the repo actually has |
|---|---|---|
| repository | `MarSaKi/BEVBert` | **`MarSaKi/VLN-BEVBert`** (the former 404s) |
| `code_root` | repo root | **`external/BEVBert/bevbert_ce`** — the VLN-CE half is a subdirectory; the root is the discrete-VLN (MatterSim) code |
| trainer | `vlnce_baselines.ss_trainer_ETP` | **`vlnce_baselines.ss_trainer_BEV`** (registered `SS-BEV`; `run_r2r/iter_train.yaml` sets `TRAINER_NAME: SS-BEV`) |
| benchmarks | R2R **and** RxR | **R2R-CE only** — README "Release VLN-CE (R2R-CE) code", no `run_rxr/`, no RxR-CE checkpoint |

Confirmed compatible with the unchanged `etpnav_worker`: `ss_trainer_BEV.RLTrainer` has
all five methods the worker calls, `_nav_gmap_variable(cur_vp, cur_pos, cur_ori)` keeps
ETPNav's signature (no `task_type`), `extract_instruction_tokens(..., pad_id=0)` likewise,
and `run_r2r/r2r_vlnce.yaml` reads `R2R_VLNCE_v1-2_preprocessed_BERTidx` — so
`dataset: {r2r: bert}` was right.

The RxR consequence is a paper edit, not a lost run: every BEVBert RxR cell is `\nmark`,
declared as `unavailable.rxr_arm` in the config, and `describe --agent bevbert --benchmark
rxr` now reports `arm_available: false` rather than resolving paths that never existed.

## What D6 still needs

1. **The CE weights.** `ckpt/ckpt.iter9600.pth` per `run_r2r/main.bash`
   (`EVAL.CKPT_PATH_DIR ckpt/ckpt.iter9600.pth`, `IL.back_algo control`,
   `ALLOW_SLIDING True` → the R2R gate arm is `slide`, same as ETPNav/ETP-R1). Hosted on
   **Google Drive** (`1-2u1NWmwpX09Rg7uT5mABo-CBTsLthGm`), wired into
   `scripts/download_checkpoints.sh --fetch`. Drive throttled the transfer at ~1 GB on
   2026-09-11 and it has not completed; re-run the script (it resumes). Plus the BEV
   pretrain weights from the second Drive link in the README.
2. Fill `published` from BEVBert Tab. 2 once the arm runs; drop `assets_pending`.
3. Smoke + `slide`/R2R gate + the R2R arms.

## Value column

`\nmark` — map-based topological planner, released checkpoint has no value head (declared
in `bevbert.yaml` `unavailable.value_estimate`).
