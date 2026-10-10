# g3D-LF  — integration notes (2026-09-26)

g3D-LF (Wang & Lee, CVPR 2025, arXiv:2411.17030) **replaces HNR** in the model set.
HNR's lookahead *navigation* model was never released: `github.com/MrZihan/HNR-VLN`
strikes out "Release the fine-tuning code / checkpoints of the Lookahead VLN Model" with
"See the improved model g3D-LF". g3D-LF is that successor, same first author, and its
panorama R2R-CE val-unseen row (Table 2) is NE 4.53 / OSR 68 / **SR 61 / SPL 52**, against
HNR's 4.42 / 67 / 61 / 51 and ETPNav's 4.71 / 65 / 57 / 49 in the same table. User
approved the swap 2026-09-26.

## What it is

`external/g3D-LF/Lookahead_3DFF` (@ `9338aa5`) is an ETPNav fork. Per decision it adds:

1. the waypoint pass writes the 12 views (CLIP ViT-L/14@336 grid features + depth) into a
   3D feature field at the agent's pose (`Feature_Fields.update_feature_fields_habitat`);
2. after ETPNav's navigation head, the **two** most probable candidates get panoramic
   features rendered from the field at their positions (`run_panorama_encode`) and are
   re-scored by a `lookahead_navigation` head; those two logits are overwritten, and the
   argmax of the fused logits is the decision;
3. eval runs under `@torch.no_grad()` + `autocast()`.

Stop scores (`gmap.node_stop_scores`) stay the pre-lookahead navigation probabilities,
as upstream: its `fused_logits = nav_logits` aliases the tensor, but `nav_probs` is
computed before the in-place overwrite.

## How the harness hosts it

| Piece | Where |
|---|---|
| worker | `vlnsafe/workers/g3dlf_worker.py` — `G3DLFDecisionEngine(ETPNavDecisionEngine)`; overrides `reset_episode` (field reset + camera setting, HFOV 90) and `decide` (the three changes above). `etpnav_worker.py` is **not** edited, so the landed ETPNav / ETP-R1 arms keep their worker byte-identical |
| registry | agent key `worker_module` picks the `-m` module (default `vlnsafe.workers.etpnav_worker`) |
| camera | `rig: {rgb: {width: 336, height: 336}}` → `sensors.apply_rgb_size` before `build_etpnav_rig`; no-op for agents without a `rig:` block (tested) |
| env | `$VLNSAFE_ENVS/vlnsafe-g3dlf` = clone of `vlnsafe-etpnav` (py3.6, torch 1.10.2+cu113, habitat 0.1.7) + `ftfy regex joblib open3d torch_kdtree tinycudann`. `tinycudann` is built with the system nvcc 11.5 and a **GCC 10.4 host compiler** from a standalone env `$VLNSAFE_ENVS/gcc10` (nvcc 11.5 + GCC 11 fails in `std_function.h`: "parameter packs not expanded"). Build scripts: `$VLNSAFE_SCRATCH/g3dlf_env/build_env{,2}.sh` |
| weights | HF dataset `MrZihanWang/g3D-LF` → `$VLNSAFE_CHECKPOINTS/g3dlf/`: `Lookahead_3DFF/data/logs/checkpoints/release_r2r/ckpt.iter25000.pth`, `Lookahead_3DFF/pretrained/r2r_ce/mlm.sap_habitat_depth/ckpts/model_step_100000.pt`, `3dff.pth`. CLIP ViT-L/14@336 is fetched by `clip.load` on first use into `~/.cache/clip` |

## Local changes to the external checkout (not upstream)

- `Lookahead_3DFF/vlnce_baselines/models/encoders/instruction_encoder.py` — **missing
  upstream** but imported by `Policy_ViewSelection_ETP.py`; copied from ETPNav (only
  `InstructionEncoder` is imported).
- `Lookahead_3DFF/bert_config` → symlink to ETPNav's.
- `Lookahead_3DFF/data/` — the same symlinks as `external/ETPNav/data/`, plus
  `data/3dff.pth` → the downloaded field weights (the trainer loads that relative path).

## Known differences from the published evaluation

- Upstream environment is habitat-sim **0.2.1**, py3.7, torch 1.13.1 (`environment.yml`);
  the harness simulates on 0.1.7 and the worker runs torch 1.10.2. Hence the reproduction
  tolerance is the VLA one, ±2 SR (`g3dlf.yaml`), not ETPNav-family ±1.
- The gate (`slide`, R2R, full split) decides whether the port holds. If it misses,
  suspects in order: autocast numerics on torch 1.10, the observation-key order the
  waypoint pass relies on ("You might need to double check the keys order"), RGB resize
  transforms at 336.

## Risk signals

`candidates`, `stop_prob`, entropy and top-2 margin are taken from the **fused**
(post-lookahead) distribution — the one the decision is the argmax of.

## Value column

`\nmark` — no value head in the released checkpoint (`unavailable.value_estimate`).
RxR: no RxR-CE lookahead checkpoint is released (`unavailable.rxr_arm`) → `\nmark`.
