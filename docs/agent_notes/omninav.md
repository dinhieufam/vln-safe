# OmniNav  — integration notes (2026-09-26)

OmniNav (Xue et al., ICLR 2026, arXiv:2509.25687). Table 1, R2R-CE val-unseen:
NE 3.74 / OS 74.6 / **SR 69.5 / SPL 66.1**; RxR-CE val-unseen NE 3.77 / SR 73.6 / SPL 62.0
(the row without the policy head: R2R SR 59.8). This is the paper's own table, so the
N35 survey figure taken from Efficient-VLN Table 1 is confirmed. Backbone:
Qwen2.5-VL-3B-Instruct. The flow-matching checkpoint is the one upstream runs by default.

## What it is

`external/OmniNav` @ `e8b4859` (2026-09-11), symlink to `$VLNSAFE_SCRATCH/omninav_repo`
(`scripts/setup_external.sh`). R2R/RxR inference lives in `infer_r2r_rxr/`:

- `agent/waypoint_agent.py::Waypoint_Agent`: per step it adds the **left, right, front**
  RGB frames to its history (at most 20 past front frames at 1/4 size, uniformly thinned;
  the first frame is always kept), builds the prompt, and runs one forward pass. The
  flow-matching head (10 denoising steps from `torch.randn` noise) returns 5 waypoints
  `(x, y)`, 5 headings `(sin, cos)` and 5 arrive values.
- `evaluate_agent`: STOP when **all five** arrive values exceed 0.5; otherwise
  `GO_TOWARD_POINT(r = |waypoint[0]|, theta = recover_angle[0])`. It also STOPs after
  **150 steps**, or after **25 steps** at which distance-to-goal did not change.
- `GO_TOWARD_POINT` (`VLN_CE/habitat_extensions/actions.py`) is VLN-CE's waypoint
  teleport: one `sim.step_filter(pos, pos + r·dir(theta))`, then the agent is rotated to
  face along `theta`. The model's waypoints are scaled by 0.3, so `r` is about 0.1–0.4 m.
- Camera (`vlnce_task_r2r_pano.yaml`): four RGB 720×640, HFOV 110, 1.25 m, front/left/
  right/back (back is never read). `ALLOW_SLIDING: True` → the published number is `slide`.

## How the harness hosts it

| Piece | Where |
|---|---|
| worker | `vlnsafe/workers/omninav_worker.py`. It imports upstream's `Waypoint_Agent` and subclasses it; `act()` is upstream's minus the PNG/GIF writes. Flow matching on, `SAVE_RANDER_IMG` off |
| execution | **the waypoint is walked out as primitives** (see below) |
| stop | upstream's rule (all 5 arrive > 0.5) and both early stops, applied per decision in the worker. The "did not move" count uses the agent's position, not the privileged distance-to-goal; the two agree except when the agent moves along a constant geodesic distance |
| rig | `sensor_rig: omninav` → `sensors.build_omninav_rig`: `rgb_front/left/right` at the config's `rig:` (720×640, HFOV 110, 1.25 m) + the benchmark's forward `DEPTH_SENSOR` (for `step_clearance_m`, as for the other VLAs; the worker never sees depth). The clearance ring is added as for every T1 agent |
| provider | `family: omninav_worker` → `WorkerActionProvider`, no occupancy, no streaming: OmniNav's history holds only the frames it decided on, which is what one request per decision gives |
| seed | flow matching samples noise, so the worker seeds torch per decision from (run seed, instruction, decision index): an episode reproduces regardless of shard |
| env | `$VLNSAFE_ENVS/vlnsafe-omninav` (`scripts/build_vla_envs.sh omninav`): py3.9, torch 2.6.0, OmniNav's vendored `transformers-main` (4.51.3; its Qwen2.5-VL carries the action heads), flash-attn 2.7.4.post1 prebuilt wheel, diffusers 0.33.1 |
| weights | ModelScope `chongchongjj/OmniNav_Flow` @ `f73a8d0` → `$VLNSAFE_CHECKPOINTS/omninav/OmniNav_Flow` (9.5 GB, bf16). One checkpoint for R2R, RxR and OVON |

### Why primitives, and how

The harness's collision predicate is habitat's `previous_step_collided`, which only
primitive actions set, and the guard vetoes forward primitives. A native
`GO_TOWARD_POINT` step would make OmniNav the only agent measured on a different action
space. So each waypoint becomes:

- `theta` snapped to the benchmark's turn unit (15° R2R, 30° RxR) with the ETPNav hop
  rule (`geometry.turn_actions_for_angle`; positive = left, as upstream);
- `round((r + residual) / 0.25)` forward steps; the remainder carries to the next decision
  of the episode, so the path length is unbiased (a 0.3 m waypoint is one 0.25 m step and
  the 0.05 m is spent later);
- a decision that rounds to no turn and no step walks one forward step (an empty chunk
  would read as STOP), paid back through the residual.

## Known differences from the published evaluation

- Action quantisation above (heading error ≤ 7.5°, distance deferred ≤ 0.125 m); the
  model re-plans from fresh views each decision, so errors do not accumulate open-loop.
- The harness's 500-primitive episode cap is the same for every agent; upstream's is 500
  waypoint steps (in practice 150, its early stop). Episodes with 150 decisions of ~3
  primitives can hit it.
- Upstream recomputes the navmesh of scene `oLBMNvg9in8` with cell height 0.1
  (`EVAL.NAVMESH_TARGET_SCENE`); the harness does not — the scene must be the same for
  every agent.
- torch 2.6 instead of the Docker image's 2.7.
- Hence tolerance ±2 SR (`omninav.yaml`). If the gate misses, suspects in order: action
  quantisation (test: forward unit 0.1 m on a 100-episode subset), the navmesh scene
  (compare SR with that scene excluded), noise seeding.

## Risk signals

No categorical over STOP/forward/left/right exists, so `prob_*` are UNAVAILABLE.
`stop_prob` = min of the five arrive values (what the stop rule thresholds); the raw five,
`r`, `theta` and the residual travel in the response `meta`.

**2026-09-29:** weights complete and verified (both shards vs safetensors headers + index total). Env: `einops` added (flash-attn runtime dep, skipped by `--no-deps`); `check_vla_envs.py` 0 problems. The `slide` reproduction gate writes directly to `results/omninav/r2r/slide`.
