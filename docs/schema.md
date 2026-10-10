# The released log contract

Every arm writes two CSVs and one JSON into
`results/<agent>/<benchmark>/<regime>/`. Column names and order are fixed in
`vlnsafe/records/schema.py`; the writer refuses any column not listed there, so a schema
change is a one-file diff. Vocabulary: agents `seq2seq`, `cma`,
`navila`, `streamvln`, `bevbert`, `etpnav`, `etpr1`; benchmarks `r2r`, `rxr`; regimes
`block` (habitat `no_sliding`) and `slide` (habitat `none`).

Two rules that have already cost this project time:

- **`UNAVAILABLE` is a value**, never a blank and never a zero. A blank in the CMA
  `internal_*` columns is exactly how a missing read-out went unnoticed for a month.
- **Every geometry constant is read from the merged benchmark config at run time** and
  echoed into `manifest.json`. R2R and RxR differ in HFOV (90 vs 63), camera height
  (1.25 vs 0.88 m) and turn angle (15° vs 30°); a hard-coded R2R value silently corrupts
  every RxR geometry number.

## `episodes.csv` — one row per episode, flushed as it completes (runs are resumable)

| Group | Columns |
|---|---|
| identity | `episode_id, scene_id, instruction_id, agent, benchmark, regime, seed, shard_index` |
| task | `success, spl, navigation_error_m, oracle_success, ndtw, sdtw, path_length_m, step_count` |
| safety, engine | `collision_count, nc_rate, collision_event_count, dirty_success_flag` |
| safety, proxy | `proxy_collision_count` — displacement-based; reported only as the object of the P2 validity result |
| clearance | `min_clearance_m, clearance_p10_m, clearance_p50_m, near_miss_count, time_in_risk_rate` |
| termination | `stuck_flag, stuck_event_count` |
| guard | `guard_abort_count, guard_yield_count` — `vlnsafe/guard.py` depth vetoes (stall vetoes are not counted here; count `guard_action == stall` in the step log) and livelock yields this episode; `UNAVAILABLE` without `--guard` |
| tryout | `tryout_escape_count` — escapes the ETP-lineage Tryout controller (`vlnsafe/agents/tryout.py`) started this episode; `UNAVAILABLE` without `--tryout` |

`collision_count` / `nc_rate` are the **physics-engine** predicate
(`env.sim.previous_step_collided`). `proxy_collision_count` is the displacement predicate
(a `MOVE_FORWARD` that moved < 1e-4 m). In the pre-release logs the naming was the
other way round (`collision_count` = proxy, `native_collision_count` = engine); the three
migrated ETPNav arms were renamed on the way in (`results/provenance.md`).

## `steps.csv` — one row per executed primitive

| Group | Columns |
|---|---|
| identity | `episode_id, step_index, is_decision, decision_index` |
| action | `proposed_action, executed_action, planned_chunk` (JSON list, VLA / T2 hops) |
| outcome | `sim_displacement_m, distance_to_goal_m, progress_delta_m, engine_collision_flag, proxy_collision_flag, step_clearance_m, near_miss_flag` |
| internal (decision rows only) | T1: `prob_stop, prob_forward, prob_left, prob_right`. T2: `candidate_entropy, candidate_entropy_norm, top2_margin, stop_prob, num_candidates, value_estimate`. `UNAVAILABLE` where an agent cannot supply one |
| external, every step | `predicted_clearance_m` -- the depth-projected clearance prediction for the *proposed* primitive, logged regardless of whether this step is a decision (T2: the worker's value; T1: `vlnsafe/clearance_predictor.py` on the 12-camera ring, since N3) |
| external, decision rows only | `min_eps_for_safe_alt, safe_alt_exists_eps10` |
| guard, every step | `guard_flagged` (0/1), `guard_action` (`pass` / `abort` / `stall` / `maneuver` / `yield` / `stop` — `stall`: a forward vetoed because one already stalled from this pose (guard config `stall`); a veto inside a chunk ending in the policy's STOP executes the STOP), `guard_clearance_m` (memory-augmented clearance of the forward step checked; `UNAVAILABLE` on turns/stop), `guard_consecutive_aborts`. All `UNAVAILABLE` without `--guard`. With the guard on, `executed_action` can differ from `proposed_action` |
| tryout, every step | `tryout_phase`: `plan` (the decision's open-loop chunk, identical to the Tryout-off harness), `escape` (side turn, probe turns, probe forwards), `resume` (turn back to the hop heading and the hop's remaining forwards), `replan` (hops re-planned from the actual pose after an escape). `UNAVAILABLE` without `--tryout` |

`candidates.jsonl` (T2 only, optional, scratch-only): one line per decision with the full
candidate list (`prob, position, is_chosen`), so P4's geometric check can be re-run
offline without a re-run.

## `manifest.json` — the reproducibility block the paper's §4 quotes

agent id + checkpoint sha256; benchmark + split + episode count; regime and the literal
habitat `allow_sliding` value; forward step size, turn angle, agent radius and height;
camera-rig summary; habitat-lab / habitat-sim / torch versions; git commit of this repo
and of each submodule; seed; shard layout; GPU indices used; start / end time.
