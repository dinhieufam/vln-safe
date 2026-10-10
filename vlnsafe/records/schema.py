"""The released log contract as explicit column lists (docs/schema.md).

What: the exact ordered columns of episodes.csv and steps.csv, plus the rule that a
missing internal value is the string UNAVAILABLE, never a blank and never a zero (a blank
in the CMA internal_* columns is how the missing read-out went unnoticed for a month).
writers.py refuses to emit a column that is not named here, so a schema change is a
one-file diff.

Who reads it: vlnsafe.records.writers, vlnsafe.rollout (row keys), tests/test_schema.py,
analysis/ (column names).

Must not import: habitat, torch, numpy.
"""

UNAVAILABLE = "UNAVAILABLE"

# One row per episode, flushed as it completes.
EPISODE_COLUMNS = (
    # identity
    "episode_id", "scene_id", "instruction_id", "agent", "benchmark", "regime",
    "seed", "shard_index",
    # task
    "success", "spl", "navigation_error_m", "oracle_success", "ndtw", "sdtw",
    "path_length_m", "step_count",
    # safety, engine
    "collision_count", "nc_rate", "collision_event_count", "dirty_success_flag",
    # safety, proxy (object of the P2 validity result only)
    "proxy_collision_count",
    # clearance
    "min_clearance_m", "clearance_p10_m", "clearance_p50_m", "near_miss_count",
    "time_in_risk_rate",
    # termination
    "stuck_flag", "stuck_event_count",
    # guard (vlnsafe/guard.py); UNAVAILABLE in a run without --guard
    "guard_abort_count", "guard_yield_count",
    # ETP-lineage Tryout (vlnsafe/agents/tryout.py); UNAVAILABLE in a run without --tryout
    "tryout_escape_count",
)

# One row per executed primitive.
STEP_COLUMNS = (
    # identity
    "episode_id", "step_index", "is_decision", "decision_index",
    # action
    "proposed_action", "executed_action", "planned_chunk",
    # outcome
    "sim_displacement_m", "distance_to_goal_m", "progress_delta_m",
    "engine_collision_flag", "proxy_collision_flag", "step_clearance_m",
    "near_miss_flag",
    # internal, decision rows only -- T1 primitive vocabulary
    "prob_stop", "prob_forward", "prob_left", "prob_right",
    # internal, decision rows only -- T2 spatial candidate set
    "candidate_entropy", "candidate_entropy_norm", "top2_margin", "stop_prob",
    "num_candidates", "value_estimate",
    # external -- predicted_clearance_m is every step (rollout.py reads it off
    # whichever primitive is about to execute, decision or not); the other two are
    # decision rows only, same as the internal columns above
    "predicted_clearance_m", "min_eps_for_safe_alt", "safe_alt_exists_eps10",
    # guard (vlnsafe/guard.py), every step; UNAVAILABLE in a run without --guard.
    # guard_clearance_m is the memory-augmented clearance of the forward step the guard
    # checked, which may differ from predicted_clearance_m (current frame only).
    "guard_flagged", "guard_action", "guard_clearance_m", "guard_consecutive_aborts",
    # ETP-lineage Tryout, every step; UNAVAILABLE in a run without --tryout. One of
    # plan / escape / resume / replan (vlnsafe/agents/tryout.py).
    "tryout_phase",
)

# steps.csv columns that carry a decision-only value; UNAVAILABLE on non-decision rows or
# where the agent cannot supply the read-out. predicted_clearance_m is NOT here even
# though it lives in the same "external" group above -- vlnsafe/rollout.py logs it every
# step (found 2026-09-19 while introspection.py/headroom.py were verified against real
# steps.csv data: 100% of rows populated, not just decision rows).
DECISION_ONLY_STEP_COLUMNS = (
    "prob_stop", "prob_forward", "prob_left", "prob_right",
    "candidate_entropy", "candidate_entropy_norm", "top2_margin", "stop_prob",
    "num_candidates", "value_estimate",
    "min_eps_for_safe_alt", "safe_alt_exists_eps10",
)


def check_row(row, columns):
    """Raise if `row` has a key not in `columns` or is missing one. Returns the row
    ordered as `columns` for the CSV writer."""
    extra = set(row) - set(columns)
    if extra:
        raise KeyError("row has columns not in the schema: " + ", ".join(sorted(extra)))
    missing = set(columns) - set(row)
    if missing:
        raise KeyError("row is missing schema columns: " + ", ".join(sorted(missing)))
    return [row[c] for c in columns]
