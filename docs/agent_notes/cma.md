# CMA — integration notes (Phase D2, 2026-09-10)

CMA (Cross-Modal Attention), the recurrent VLN-CE baseline, is the paper's **T1 comparator**
for P2/P3/P4. It runs **in-process** in `vlnsafe-harness` (`family: vlnce_policy`) — VLN-CE's
`vlnce_baselines` is already importable there, so no subprocess.

## Provider

`vlnsafe/agents/vlnce_policy.py::VlncePolicyProvider`, ported from
the pre-release harness's CMA action provider with the **read-out fix**:

- **Interface:** `reset / act / planned_chunk / update_after_step`, plus the
  `pending_decision_signals` attribute rollout reads then clears. A T1 policy decides one
  primitive per step, so `planned_chunk()` is always `[]` and every step is a decision.
- **Action:** `act()` inlines `ILPolicy.act(deterministic=True)` — one RNN forward
  (`policy.net(...)` → `policy.action_distribution(features)` → `distribution.mode()`).
  Byte-identical to the archived `policy.act(..., deterministic=True)`, so the Phase E CMA
  reproduction is unaffected; the categorical is now captured in the same pass.
- **Read-out (D2):** `distribution.probs[0]` → `prob_stop / prob_forward / prob_left /
  prob_right` (habitat action order: STOP 0, FORWARD 1, LEFT 2, RIGHT 3;
  `CategoricalNet` `dim_actions == 4`). the pre-release provider never populated
  `pending_decision_signals`, which is why every `internal_*` column of the archived CMA
  log is empty. `rollout.py` now writes these on decision rows; entropy / margin / stop
  probability are derived offline in `analysis/`.

## Env

`sensor_rig: vlnce_recurrent` — `build_r2r_config` keeps VLN-CE's own single RGB+DEPTH rig
from `cma_pm_da_aug_tune.yaml` untouched (the archived autopsy ran `keep_sensors=True`) and
only pins `DATASET.DATA_PATH` to the resolved root. `dataset: {r2r: official}` — CMA reads
the untokenised split and tokenises internally. `step_clearance_m` comes from the single
`depth` sensor (the panoramic clearance ring is not built for this rig), matching the
archived run.

## Verified (2026-09-10, `vlnsafe-harness`)

- torch 1.10.2+cu113, CUDA available.
- `CMAPolicy` registered; has `act`, `build_distribution`, `from_config`;
  `ILPolicy.__init__` sets `action_distribution`.
- `best_checkpoint.pth` loads: 520 params, `{state_dict}`.

## RxR: the checkpoint exists (2026-09-11)

Our initial plan listed "no public RxR-CE checkpoint for CMA/Seq2Seq" as a risk whose
default decision was to make those cells `\nmark`. **That is wrong for CMA**: the
RxR-Habitat Challenge's official *Monolingual CMA Baseline* is public — VLN-CE's README
links en/hi/te weights at 196 MB each. The English one is downloaded and verified
(`$VLNSAFE_CHECKPOINTS/cma/rxr_cma_en/ckpt.pth`, 519 params, sha256 in
`scripts/download_checkpoints.sh`), and `cma.yaml` now carries `checkpoint.rxr` +
`exp_config.rxr: vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml`.

So **PLAN §7 run 4b (`cma/rxr/{block,slide}`) is a real arm**, not an `\nmark`. There is
still no *published* val-unseen SR to gate against (the challenge reports on its
leaderboard), so `published.rxr.sr` stays `null` — the arm runs, it just has no
reproduction gate. Seq2Seq RxR remains genuinely unavailable.

One caveat before running it: `build_rxr_config` registers a pass-through
`RxRInstructionSensor` that hands the raw XLM-R tokens through, because ETPNav's
checkpoint consumes them directly. CMA wants VLN-CE's *real* RxR instruction sensor
(precomputed BERT text features, never downloaded). Sort that out in the run, not before.

## Owed by the D2 GPU arms (not code)

- 5-episode R2R smoke, then **full** R2R val-unseen × {block, slide} (removes the `†` from
  Tables 3 and 6 — `episode_subset: {r2r: 700}` in the config is cleared once these land).
- `cma/rxr/{block,slide}` once the instruction-sensor question above is settled.

## Value column

`\nmark`: CMA is imitation-learned and has no value head (declared in `cma.yaml`
`unavailable.value_estimate`).
