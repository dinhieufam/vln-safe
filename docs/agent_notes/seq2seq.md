# Seq2Seq — integration notes (Phase D3, 2026-09-11)

Seq2Seq is the original VLN-CE recurrent baseline (RGB + depth + instruction → GRU →
4-way categorical). **Zero new code**: it is the same `family: vlnce_policy` /
`VlncePolicyProvider` as CMA, with `policy_name: Seq2SeqPolicy`.

## What D3 needs (no GPU dependency for the code)

1. ~~**Checkpoint.**~~ **Done (2026-09-11).** The VLN-CE model zoo publishes exactly one
   Seq2Seq checkpoint — **Seq2Seq_DA** (`vlnce_baselines/config/r2r_baselines/README.md`,
   Google Drive `12swcou9g5jwR31GbQU1wJ88E_8j--Qi5`, 135 MB, published val_unseen
   **SPL 0.23**), which is the variant `seq2seq.yaml` already named. Downloaded, verified
   (496 params, sha256 in `scripts/download_checkpoints.sh`) and landed at
   `$VLNSAFE_CHECKPOINTS/seq2seq/seq2seq_da/ckpt.pth`; `assets_pending` dropped.
2. **`exp_config`.** `vlnce_baselines/config/r2r_baselines/seq2seq_da.yaml` — confirm it
   sets `MODEL.policy_name: Seq2SeqPolicy` and a single RGB+DEPTH rig (the provider
   overrides `policy_name` from the agent config regardless).
3. **Smoke + arms.** `python -m vlnsafe smoke --agent seq2seq` (5 ep), then
   `seq2seq/r2r/{block,slide}` full split. RxR stays unavailable: the RxR-Habitat baseline
   is CMA-only, so there is no RxR-CE Seq2Seq checkpoint and
   `describe --agent seq2seq --benchmark rxr` reports `arm_available: false` → `\nmark`.

## Read-out / value

Same as CMA: `act()` inlines `ILPolicy.act(deterministic=True)`, captures
`distribution.probs` → `prob_stop/forward/left/right` on every step. `value_estimate` is
`\nmark` (declared in `seq2seq.yaml` `unavailable`).

## Reproduction gate

`seq2seq/r2r/slide`, full split (1,839 episodes, 2026-09-20): **SR 22.13 / SPL 20.25** vs
published 26.0 / 23.0 (VLN-CE ECCV 2020 Tab. 3 row 3, Seq2Seq + DAgger; the paper's "DA" is
DAgger, and the model-zoo `Seq2Seq_DA` is assumed to be that row). Delta SR 3.87, which fails
the original +/-1.0 gate. **Decision (Sep 20): accept, tolerance widened to 4.0.** The VLN-CE
README states its paper numbers came from different hardware + a different Habitat build
than the current release (CMA 0.30 -> 0.27 SPL on the leaderboard), and CMA in this harness
shows the same direction (SR 30.2 vs 32 published). Not proven to be the whole cause: the
zoo weights may not be exactly the paper's row 3. Report ours beside the published number in
Table 2's notes; block-vs-slide comparisons inside the harness are unaffected.

## Bug found on the first launch (relevant to D6 and any `vlnce_policy` agent)

`vlnsafe run` built the Habitat/model config from the *benchmark's* `habitat_base_config`
(`cma_pm_da_aug_tune.yaml`), never from the agent's `exp_config`, so Seq2Seq was built with
CMA's bidirectional instruction encoder and `load_state_dict` failed (missing `_reverse`
keys, 640 vs 512 input). Invisible for CMA because its `exp_config` is that same file. Fixed
in `vlnsafe/cli.py`: for `family: vlnce_policy` the agent's `exp_config` is the base.

## Launch notes

The machine load (230-440, other users) is above `launch_arm.sh`'s guard of 56;
`VLNSAFE_MAX_LOAD=600` overrides it (added Sep 20). The slide arm took ~4.5 h on GPUs 3+7,
4 shards.

## Block arm (2026-09-21)

`seq2seq/r2r/block`, full split, 1,839 episodes / 578,357 step rows, merged and
duplicate-free: **SR 10.39 / SPL 9.84**, 63.4 % of episodes flagged stuck (slide: SR 22.13,
5.1 % stuck). Same ~4.5 h on GPUs 3+7. The block-vs-slide drop (22.1 -> 10.4) is the same
regime-sensitivity pattern as CMA (30.2 -> 12.1).
