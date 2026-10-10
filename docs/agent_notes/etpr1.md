# ETP-R1 — integration notes (Phase A, read 2026-09-10)

Read from `github.com/Cepillar/ETP-R1` (pinned in `scripts/setup_external.sh`) against
`external/ETPNav` at the commit used for our ETPNav arms.

## 1. Does the rollout still fit `workers/etpnav_worker.py`? Yes, as a config swap.

`vlnce_baselines/ss_trainer_ETP_R1.py` is registered `SS-ETP-R1` and is the trainer used at
**eval** time for the released GRPO checkpoints (not `GRPO_trainer_ETP_R1.py`, which is
training-only — RL fine-tuning — and has no `_eval_checkpoint`/`inference` method; ETP-R1
reuses the SS trainer for inference exactly as ETPNav does). Diffed `_eval_checkpoint` +
`inference` (ETPNav `ss_trainer_ETP.py:644-900` vs ETP-R1 `ss_trainer_ETP_R1.py:743-1000`,
257 vs 258 lines): near-identical, differences are whitespace plus:

- an instruction `task_type` tag (1=r2r, 2=rxr) threaded through
  `extract_instruction_tokens(..., task_type=...)` and `_nav_gmap_variable(..., task_type)` —
  new required kwarg, not present in ETPNav's signature. **Worker action:** the ETPNav worker's
  call sites for these two functions need `task_type` passed when `family: etpnav_worker` and
  `id: etpr1`; make it a config field (`configs/agents/etpr1.yaml`), not a hard-coded literal,
  same rule as the R2R/RxR rig constants.
- `not_done_index`-based slicing of `txt_masks`/`txt_embeds` is dropped in ETP-R1 (indexes the
  full batch instead) — behaviourally equivalent for a single-episode-per-shard worker; only
  matters if a shard ever batches multiple envs per forward pass, which ours doesn't.

The four symbols the worker's decision engine reads —
`nav_outs["global_logits"]`, `gmap.node_stop_scores`, `gmap.ghost_aug_pos`,
`config.MODEL.consume_ghost` — are present **unchanged** in ETP-R1 (`graph_utils.py`,
`models/etp/ETP_R1_vilmodel_cmt.py`). `back_algo` is read from `config.IL.back_algo` at eval
time in both repos identically (`main_server.bash` sets `IL.back_algo control` for eval, same
value ETPNav uses); the `config.GRPO.back_algo` path some code greps turn up is
`GRPO_trainer_ETP_R1.py`'s own training-time rollout and is not on the eval path.

**Conclusion: subclass, not a fork.** `workers/etpnav_worker.py` serves ETP-R1 via
`family: etpnav_worker`, `code_root: external/ETP-R1`, plus the one new `task_type` field.

## 2. Sensor rig — unchanged.

`habitat_extensions/obs_transformers.py` is **byte-identical** to ETPNav's (`diff` empty).
`run_r2r/r2r_vlnce.yaml`: RGB 224×224 HFOV 90, Depth 256×256 HFOV 90, forward 0.25 m, turn
15°, `ALLOW_SLIDING: True` — matches ETPNav's R2R rig exactly. `run_rxr/rxr_vlnce.yaml`: HFOV
63, turn 30°, height 0.88 m, radius 0.18 m, `ALLOW_SLIDING: False` — matches ETPNav's RxR rig
exactly. Pretrain is `mlm.sap_habitat_depth` (confirmed, see checkpoints below), depth-only
variant, consistent with the 12-view rig. **`sensor_rig: etpnav_12view` applies unchanged.**

## 3. Value head — confirmed absent, `\nmark` is correct.

`vlnce_baselines/models/R1Policy.py` defines a `Critic(nn.Module)` class (a 768→512→1 MLP),
but it is **never instantiated** anywhere in the repo (`grep -rn "Critic("` finds only the
class statement) and `GRPO_trainer_ETP_R1.py` never references `critic` — GRPO's
group-relative baseline replaces a learned value function by construction, and this class
looks like dead code carried over from an earlier ablation. Loaded the released checkpoint
state dict directly (937 keys) and confirmed zero `critic`/`state2value`/`value_head` keys.

**Decision:** `Value` column is `\nmark` in every ETP-R1 row,
declared in `configs/agents/etpr1.yaml`'s `unavailable:` block with reason "GRPO is
critic-free; the released checkpoints expose no value head (confirmed: state dict has no
critic/value parameters, and the repo's unused `Critic` class is never instantiated)."

## 4. Regime named in config — yes, and RxR differs from R2R.

- R2R (`run_r2r/r2r_vlnce.yaml` + `habitat_extensions/config/r2r_vlnce.yaml`):
  `ALLOW_SLIDING: True`, reinforced by an explicit
  `TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True` override in both
  `main_server.bash` and `auto_eval_server.bash`. **R2R gate arm is `slide`**, matching the
  published SR 65 / SPL 56.
- RxR (`run_rxr/rxr_vlnce.yaml`, and all three `habitat_extensions/config/rxr_vlnce_{en,hi,te}.yaml`):
  `ALLOW_SLIDING: False`. **ETP-R1's own RxR eval regime is `block`**, not `slide`.
  `configs/agents/etpr1.yaml` leaves `published.rxr.sr: null` (no public RxR number to
  gate against), so this doesn't block a reproduction gate, but the `block` arm is the one to
  treat as "closest to their setup" if we ever want an informal sanity check before the full
  four-arm matrix.

## Extra finding: R2R dataset variant, not the BERTidx one ETPNav uses

ETP-R1's dataset config is `TYPE: VLN-CE-v2`, `DATA_PATH: data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/...`
— **XLM-RoBERTa tokenization**, not the BERT-tokenized `R2R_VLNCE_v1-2_preprocessed_BERTidx`
ETPNav reads. This is bundled inside `extra_files.zip` (not something we need to source
separately) and has already been unpacked to
`$VLNSAFE_SCRATCH/etpr1_vendor_dump/data/datasets/{R2R_VLNCE_v1-3_preprocessed_xlmr,RxR_VLNCE_v0_enc_xlmr}`.

**The RxR_VLNCE_v0_enc_xlmr name collides but the content differs**: ETP-R1's copy (447 MB —
raw guide/follower files, segmentation scripts) is not the same dataset as the 28 MB
`RxR_VLNCE_v0_enc_xlmr` in `$VLNSAFE_DATASETS/` from the
ETPNav store (`diff -rq` shows disjoint file sets, not just different content of shared
files). **Open question for Phase C**: `configs/benchmarks/r2r.yaml` / `rxr.yaml` cannot
name a single dataset path per benchmark if agents disagree on tokenizer — needs either a
per-agent dataset override in `configs/agents/etpr1.yaml`, or a benchmark config that reads
the encoding from the agent config. Decide in Phase C, not now; this is exactly the kind of
per-agent literal §2's naming rules want kept out of the harness. The two ETP-R1-specific
dataset copies stay in `scratch/etpr1_vendor_dump` (not `datasets/`) until that decision is
made, so they don't silently shadow the ETPNav ones.

## Checkpoints (verified, `scripts/download_checkpoints.sh`)

| File | sha256 | Final path |
|---|---|---|
| `ckpt.iter270.pth` (R2R GRPO) | `8f90cebb...c54f61c` | `checkpoints/etpr1/release_r2r_grpo/store/ckpt.iter270.pth` |
| `ckpt.iter1320.pth` (RxR GRPO) | `3796c9c9...48f27821` | `checkpoints/etpr1/release_rxr_grpo/store/ckpt.iter1320.pth` |
| `model_step_367500.pt` (pretrain) | `203fe62c...eab6c3cf` | `models/etpr1/mlm.sap_habitat_depth/store2/model_step_367500.pt` |

`main_server.bash`'s own `EVAL.CKPT_PATH_DIR` / `INFERENCE.CKPT_PATH` both point at
`release_r2r_grpo/store/ckpt.iter270.pth`, confirming this is the right file (not one of the
`*_dagger` intermediate-training checkpoints, kept at
`checkpoints/etpr1/dagger_stage/` for reference only — not part of any paper arm).

## Phase D1 resolution (2026-09-10)

**Worker serves ETP-R1 as a config swap — landed and interface-verified under `vlnsafe-etpnav`.**

1. **No fifth conda env.** `vlnsafe-etpnav` is py3.6.13 / torch **1.10.2+cu113** /
   transformers 4.12.5. ETP-R1's `environment.yaml` pins torch 1.9.1, but
   `ckpt.iter270.pth` (R2R GRPO) `torch.load`s cleanly under 1.10.2 — top-level
   `{state_dict, config}`, **937 params**, `net.module.` (DataParallel) prefix, and a
   `vln_bert.embeddings.task_type_encoding.weight` table. `import vlnce_baselines` from
   `external/ETP-R1` registers `SS-ETP-R1` with no missing deps.
2. **`\nmark` re-confirmed.** Zero `critic` / `value_head` / `v_head` / `state2value`
   keys in the 937-param state dict (matches A4 §3).
3. **Worker wiring** (`vlnsafe/workers/etpnav_worker.py`, `vlnsafe/agents/registry.py`):
   - `--trainer-module` selects the SS trainer class by import path. `etpnav.yaml` →
     `vlnce_baselines.ss_trainer_ETP`; `etpr1.yaml` → `vlnce_baselines.ss_trainer_ETP_R1`
     (both configs carry `worker_trainer_module:`; the worker default is the ETPNav one).
   - `--task-type` (1=r2r, 2=rxr) is emitted by `registry.etpnav_worker_argv` **iff** the
     agent config has a `task_type:` block — ETPNav has none, so its call sites are
     untouched. Verified against the live signatures: `extract_instruction_tokens(...,
     task_type=…)` (6th param, keyword) and `RLTrainer._nav_gmap_variable(self, cur_vp,
     cur_pos, cur_ori, task_type)` (4th positional).
   - When `task_type` is set the engine also pins `instr_pad_id = 1` for both benchmarks
     (`ss_trainer_ETP_R1.rollout` hard-codes it — XLM-R pad id), vs ETPNav's
     task-dependent 0/1.
4. **Instruction-file suffix generalised.** `build_r2r_config` no longer hard-codes
   `{split}_bertidx.json.gz`; `_instruction_data_path(instr_data_root, split)` probes
   `{split}_bertidx.json.gz` then `{split}.json.gz` under the resolved root. Byte-identical
   for ETPNav (bert root → `_bertidx`); ETP-R1's xlmr root → `{split}.json.gz`.
   `pytest tests/test_agent_wiring.py` locks all of the above.

**Still owed by the D1 GPU arm (not code):**

- `external/ETP-R1/data/wp_pred/check_cwp_bestdist_hfov90` must resolve (waypoint
  predictor weights, `ss_trainer_ETP_R1.py:214`) — wire the `external/ETP-R1/data`
  symlink / `$VLNSAFE_MODELS/wp_pred` before the first live run.
- 5-episode R2R `slide` smoke, then the full-split reproduction gate (SR within 1.0 of
  published 65).

## Phase C resolution (2026-09-10)

**`configs/agents/etpr1.yaml` written**, `family: etpnav_worker`, `code_root: external/ETP-R1`
(symlink -> `scratch/etpr1_repo`, matching the other three; D1 pins it as a submodule),
`sensor_rig: etpnav_12view`, `back_algo: control`. `python -m vlnsafe describe --agent etpr1
--benchmark {r2r,rxr}` resolves every path and all exist. Value column declared `\nmark`
via `unavailable.value_estimate` (question 3).

**`task_type` (question 1)** is a config field: `task_type: {r2r: 1, rxr: 2}`. The
`etpnav_worker` provider passes it to `extract_instruction_tokens` / `_nav_gmap_variable`
when present; ETPNav's config omits it, so its call sites are unchanged. Wiring lands with
the D1 worker change.

**`exp_config`**: `run_r2r/iter_train.yaml` / `run_rxr/iter_train.yaml` (present in the clone;
the worker also loads `run_r2r/r2r_vlnce.yaml` transitively, training-only fields ignored).

**Dataset tokenizer split — decided.** The benchmark config enumerates dataset roots by
encoding (`configs/benchmarks/r2r.yaml` `dataset: {official, bert, xlmr}`;
`rxr.yaml` `dataset: {xlmr, base}`); the agent config names one per benchmark
(`etpr1.yaml` `dataset: {r2r: xlmr, rxr: xlmr}`, `etpnav.yaml` `{r2r: bert, rxr: xlmr}`).
`config.agent_dataset_key()` / `config.benchmark_dataset_root()` resolve it; a VLA with no
`dataset:` block falls back to `official`. This keeps the tokenizer choice out of the
harness as a literal (§2 naming rules).

- **R2R**: `R2R_VLNCE_v1-3_preprocessed_xlmr` (256 MB) promoted from
  `scratch/etpr1_vendor_dump/` into `datasets/` — no name collision with the BERTidx or the
  plain v1-3 set.
- **RxR**: no new data. The A4 note's "disjoint file sets" was the *full* trees (train /
  follower / segmentation scripts); the two files the harness reads for `val_unseen`
  (`val_unseen_guide.json.gz`, `val_unseen_guide_gt.json.gz`) are **byte-identical** (md5
  `6ff90887…`, `1b562664…`) between ETPNav's `datasets/RxR_VLNCE_v0_enc_xlmr` and ETP-R1's
  vendor copy. Both configs point `rxr` at the same root. The vendor dump stays in
  `scratch/` as history.

## Phase D1 resolution, part 2 (2026-09-17) — the v1/v2 loader gap, closed

The "Open question for Phase C" above was answered halfway: the dataset **path** resolves
correctly (`xlmr` key), but nothing carried the matching **TYPE** into
`TASK_CONFIG.DATASET`, so the Sep 11 launch died with `KeyError: 'instruction_vocab'` in
VLN-CE's v1 loader. Fixed with three changes, all in harness code, none
in the worker's model logic:

1. `configs/benchmarks/r2r.yaml` gets a `dataset_type:` map (`xlmr: VLN-CE-v2`, others
   default `VLN-CE-v1`); `vlnsafe/config.py::benchmark_dataset_type()` resolves it;
   `simulator/environment.py::build_r2r_config` threads it into
   `TASK_CONFIG.DATASET.TYPE`.
2. **`VLN-CE-v2` was not registered anywhere the harness process could see.** It only
   exists in ETP-R1's own `habitat_extensions/task.py`, which runs in a separate conda
   env/subprocess (the worker), and that version needs a `DATASET.SUFFIX` config field our
   shared `vlnce_baselines` schema does not define. Rather than pull ETP-R1's config schema
   into the harness, `simulator/environment.py::_register_xlmr_dataset()` registers a
   `VLN-CE-v2` class that is V1's loader (`external/VLN-CE/habitat_extensions/task.py`)
   with the `instruction_vocab` parse removed — checked against
   `R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz`: same episode schema as
   V1 (`episode_id, trajectory_id, scene_id, start_position, start_rotation, info, goals,
   instruction{instruction_text, instruction_tokens}, reference_path`), just missing the
   top-level `instruction_vocab` block. Verified standalone: loads all 1839 R2R val-unseen
   episodes with correct scene/instruction resolution.
3. **A second, independent bug the smoke test caught**:
   `workers/etpnav_worker.py::reset_episode` computed `all_txt_ids` but never extracted or
   passed `txt_task_encoding` to `policy.net(mode="language", ...)`, so ETP-R1's own
   `forward_txt` (which requires it once `task_type` is set) got `None` and crashed in
   `nn.Embedding`. Confirmed against ETP-R1's own `ss_trainer_ETP_R1.py::rollout()`, which
   reads `batch['txt_task_encoding']` — the same key `extract_instruction_tokens` already
   populates on our side when `task_type` is passed. Fixed by extracting and forwarding
   that key when `self.task_type is not None` (no change to ETPNav/BEVBert, which never set
   `task_type`).

**5-episode R2R smoke passes**: `data/scratch/smoke/etpr1/r2r/episodes.csv`, rc=0, 5 rows
with plausible, varying success/SPL/nDTW/navigation_error — not a crash-masked degenerate
result.

**Reproduction gate passed (2026-09-17, 16:39 UTC).** `scripts/launch_arm.sh etpr1 r2r
slide 2` (GPUs 3+4, `pick_gpus.sh 3 --exclude-egl-broken`), full R2R val-unseen split
(920 + 919 episodes), merged to `results/etpr1/r2r/slide/episodes.csv` (1,839 episode
rows, 168,578 step rows). `scripts/check_reproduction.py`: reproduced SR 64.60 / SPL 55.54
vs published SR 65.00 (ETP-R1 Tab. 2, R2R-CE val-unseen) — delta 0.40, tolerance ±1.0,
**PASS**. ETP-R1 stays the T2 headline; the BEVBert fallback does not trigger.

**All four D1 arms now landed (2026-09-18, 00:20 UTC).** Same launch pattern each time
(`scripts/launch_arm.sh etpr1 <benchmark> <regime> 2`, GPUs picked fresh per launch,
1 shard each on 2 GPUs), merged with `scripts/merge_shards.py`:

| Arm | Episodes | SR | SPL | Note |
|---|---|---|---|---|
| `r2r/slide` | 1,839 | 64.60 | 55.54 | reproduction gate, PASS vs published 65.00 (delta 0.40) |
| `r2r/block` | 1,839 | 50.52 | 43.35 | expected drop vs `slide`, no published comparator |
| `rxr/slide` | 1,223 | 64.43 | 54.43 | not gated — `published.rxr.sr` is `null` in `configs/agents/etpr1.yaml` |
| `rxr/block` | 1,223 | 24.86 | 22.00 | steep drop vs `rxr/slide`; checked against archived ETPNav `rxr/{block,slide}` (SR 23.39 vs 57.97) — same magnitude, so this is a real RxR regime-sensitivity effect, not a code bug |

D1 is done. Next for this agent: nothing — it feeds Phase E (`collect.py`/`make_tables.py`)
once that phase starts.
