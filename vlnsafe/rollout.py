"""Run one episode: reset -> until done { ask the provider for the next primitive,
refilling its planned chunk when empty; measure clearance; step the simulator; log the
step row; bookkeeping } -> write the episode row.

What: the only place in the harness that steps the simulator. No risk gate, no correction
manager, no conditions — every one of those branches from the pre-release run_episode is gone
here (its "Base" condition, which is all this reproduces, made them no-ops anyway). The
one intervention point is the optional `guard` (vlnsafe/guard.py), which may replace the
proposed primitive before it executes; without it, executed == proposed on every step.
Each logged quantity carries a comment naming the paper metric it feeds.

Who reads it: vlnsafe.cli, one call per episode in a shard.

Imports habitat only for HabitatSimActions / the metric names, inside run_episode. Must
not import torch.

Ported from the pre-release harness, Base path only:
proposal + execution + logging + clearance/collision/stuck bookkeeping; every risk/
correction/condition branch deleted.
"""
import json
import time

import numpy as np

from .geometry import compute_step_clearance_m
from .guard import GUARD_OFF_ROW
from .records.schema import UNAVAILABLE
from .simulator.pose import agent_pose, agent_position, action_to_step

_NEAR_MISS_M = 0.15   # step_clearance below this, with no engine contact -> a near miss
_IN_RISK_M = 0.25     # step_clearance below this -> a "time in risk" step
_STUCK_STREAK = 5     # consecutive blocked forward steps that count as stuck


def _num(value):
    """A metric habitat may hand back as '' -> UNAVAILABLE, else the float."""
    if value == "" or value is None:
        return UNAVAILABLE
    return float(value)


_R2R_CLEARANCE_KEYS = ["depth"] + ["depth_{}".format(30 * k) for k in range(1, 12)]


def run_episode(env, episode_meta, provider, episode_writer, step_writer,
                agent, benchmark, regime, seed, shard_index,
                clearance_sensor_keys=None, clearance_predictor=None, guard=None,
                timing_writer=None):
    """`clearance_predictor(obs, primitive) -> metres or None` fills predicted_clearance_m
    for providers that do not supply it (T1). A provider's own value stays authoritative:
    the predictor is consulted only when the provider left it None, and cli builds it only
    for providers that send no occupancy cloud, so the landed T2 columns cannot move.
    `guard` (vlnsafe.guard.GuardLayer or None) sees every step's proposal and decides what
    executes; None leaves every step as before and logs its columns as UNAVAILABLE.
    `timing_writer` (a csv.writer or None; `run --timing`) gets one row of
    wall-clock seconds per step: policy, logged predictor, guard, simulator, whole step.
    Timing only reads the clock, so the step and episode rows are unchanged."""
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    clearance_sensor_keys = clearance_sensor_keys or _R2R_CLEARANCE_KEYS

    move_forward = int(HabitatSimActions.MOVE_FORWARD)
    action_names = {
        int(HabitatSimActions.STOP): "stop",
        move_forward: "move_forward",
        int(HabitatSimActions.TURN_LEFT): "turn_left",
        int(HabitatSimActions.TURN_RIGHT): "turn_right",
    }

    episode_id = str(episode_meta["episode_id"])
    obs = env.reset()
    assert str(env.current_episode.episode_id) == episode_id, (
        env.current_episode.episode_id, episode_id)
    provider.reset(obs)
    if guard is not None:
        guard.reset()
    # The agent's own Tryout controller runs inside the provider; the
    # rollout only logs which phase each step belongs to.
    tryout = bool(getattr(provider, "tryout", False))
    # With --tryout --guard the guard checks Tryout's forwards inside the provider
    tryout_guard = getattr(provider, "tryout_guard", None) if tryout else None
    counts = guard if guard is not None else tryout_guard

    proxy_collision_count = 0       # displacement proxy (forward step, <1e-4 m)
    engine_collision_count = 0      # physics engine (previous_step_collided) -> nc_rate
    collision_event_count = 0       # engine-collision streak starts
    was_in_collision = False
    consecutive_no_progress = 0
    max_consecutive_no_progress = 0
    stuck_event_count = 0
    was_in_stuck_streak = False
    clearance_values_m = []         # -> min/p10/p50 clearance, time_in_risk_rate
    near_miss_count = 0
    time_in_risk_steps = 0
    previous_distance = None
    step_index = 0
    decision_index = -1

    while not env.episode_over:
        t0 = time.perf_counter()
        pose = agent_position(env)
        proposed = provider.act(obs, pose)
        t_policy = time.perf_counter()

        signals = getattr(provider, "pending_decision_signals", None)
        if signals is not None:
            provider.pending_decision_signals = None
            decision_index += 1
        step_pred_clearance = getattr(provider, "pending_step_predicted_clearance_m", None)
        if step_pred_clearance is None and clearance_predictor is not None:
            step_pred_clearance = clearance_predictor(obs, proposed)  # obs = pre-step frame
        t_predictor = time.perf_counter()
        planned_chunk = [action_names.get(a, a) for a in provider.planned_chunk()]
        tryout_phase = (provider.pending_tryout_phase if tryout else UNAVAILABLE)

        executed = proposed
        guard_row = GUARD_OFF_ROW
        if guard is not None:
            executed, guard_row = guard.step(obs, proposed, agent_pose(env), provider)
        elif tryout_guard is not None:
            guard_row = provider.pending_guard_row   # Vetoes ran inside Tryout
        t_guard = time.perf_counter()
        before = agent_position(env)
        obs = env.step(action_to_step(executed))
        done = env.episode_over
        provider.update_after_step(obs, executed, done)
        t_env = time.perf_counter()
        after = agent_position(env)

        displacement = float(np.linalg.norm(after - before))
        forward_blocked = executed == move_forward and displacement < 1e-4
        engine_collision = bool(env.sim.previous_step_collided)
        if engine_collision:
            engine_collision_count += 1
            if not was_in_collision:
                collision_event_count += 1
        was_in_collision = engine_collision

        if forward_blocked:
            proxy_collision_count += 1
            consecutive_no_progress += 1
        else:
            consecutive_no_progress = 0
        max_consecutive_no_progress = max(max_consecutive_no_progress, consecutive_no_progress)
        in_stuck_streak = consecutive_no_progress >= _STUCK_STREAK
        if in_stuck_streak and not was_in_stuck_streak:
            stuck_event_count += 1
        was_in_stuck_streak = in_stuck_streak

        step_clearance_m = compute_step_clearance_m(obs, sensor_keys=clearance_sensor_keys)
        near_miss_flag = 0
        if step_clearance_m is not None:
            clearance_values_m.append(step_clearance_m)
            if step_clearance_m < _IN_RISK_M:
                time_in_risk_steps += 1
            if step_clearance_m < _NEAR_MISS_M and not engine_collision:
                near_miss_count += 1
                near_miss_flag = 1

        metrics = env.get_metrics()
        distance_to_goal = metrics.get("distance_to_goal", "")
        progress_delta = ""
        if previous_distance is not None and distance_to_goal != "":
            progress_delta = float(previous_distance) - float(distance_to_goal)
        if distance_to_goal != "":
            previous_distance = distance_to_goal

        is_decision = int(signals is not None)
        sig = signals or {}
        row = {
            "episode_id": episode_id,
            "step_index": step_index,
            "is_decision": is_decision,
            "decision_index": decision_index if decision_index >= 0 else UNAVAILABLE,
            "proposed_action": action_names.get(proposed, proposed),
            "executed_action": action_names.get(executed, executed),
            "planned_chunk": json.dumps(planned_chunk),
            "sim_displacement_m": displacement,
            "distance_to_goal_m": _num(distance_to_goal),
            "progress_delta_m": _num(progress_delta),
            "engine_collision_flag": int(engine_collision),
            "proxy_collision_flag": int(forward_blocked),
            "step_clearance_m": UNAVAILABLE if step_clearance_m is None else step_clearance_m,
            "near_miss_flag": near_miss_flag,
            # T1 primitive-vocabulary read-out (vlnce_policy fills these on every step;
            # an ETPNav-family worker's signals dict has no prob_* keys -> UNAVAILABLE).
            "prob_stop": sig.get("prob_stop", UNAVAILABLE) if is_decision else UNAVAILABLE,
            "prob_forward": sig.get("prob_forward", UNAVAILABLE) if is_decision else UNAVAILABLE,
            "prob_left": sig.get("prob_left", UNAVAILABLE) if is_decision else UNAVAILABLE,
            "prob_right": sig.get("prob_right", UNAVAILABLE) if is_decision else UNAVAILABLE,
            # T2 candidate-set read-out (decision rows only).
            "candidate_entropy": sig.get("candidate_entropy", UNAVAILABLE) if is_decision else UNAVAILABLE,
            "candidate_entropy_norm": sig.get("candidate_entropy_normalized", UNAVAILABLE) if is_decision else UNAVAILABLE,
            "top2_margin": sig.get("top2_margin_prob", UNAVAILABLE) if is_decision else UNAVAILABLE,
            "stop_prob": sig.get("stop_prob", UNAVAILABLE) if is_decision else UNAVAILABLE,
            "num_candidates": sig.get("num_valid_candidates", UNAVAILABLE) if is_decision else UNAVAILABLE,
            "value_estimate": UNAVAILABLE,  # GRPO/ETPNav: no value head
            "predicted_clearance_m": UNAVAILABLE if step_pred_clearance is None else step_pred_clearance,
            "min_eps_for_safe_alt": sig.get("min_eps_for_safe_alt", UNAVAILABLE) if is_decision else UNAVAILABLE,
            "safe_alt_exists_eps10": sig.get("headroom_alt_exists_eps10pct", UNAVAILABLE) if is_decision else UNAVAILABLE,
        }
        row.update(guard_row)
        row["tryout_phase"] = tryout_phase
        step_writer.write(row)
        if timing_writer is not None:
            timing_writer.writerow([
                episode_id, step_index, row["executed_action"], row["guard_action"],
                t_policy - t0, t_predictor - t_policy, t_guard - t_predictor,
                t_env - t_guard, time.perf_counter() - t0])
        step_index += 1

    metrics = env.get_metrics()
    step_count = int(metrics.get("steps_taken", step_index))
    success = metrics.get("success", "")
    dirty_success = int(bool(success) and engine_collision_count > 0)
    episode_writer.write({
        "episode_id": episode_id,
        "scene_id": episode_meta.get("scene_id", ""),
        "instruction_id": episode_meta.get("instruction_id", "") or "",
        "agent": agent,
        "benchmark": benchmark,
        "regime": regime,
        "seed": seed,
        "shard_index": shard_index,
        "success": _num(success),
        "spl": _num(metrics.get("spl", "")),
        "navigation_error_m": _num(metrics.get("distance_to_goal", "")),
        "oracle_success": _num(metrics.get("oracle_success", "")),
        "ndtw": _num(metrics.get("ndtw", "")),
        "sdtw": _num(metrics.get("sdtw", "")),
        "path_length_m": _num(metrics.get("path_length", "")),
        "step_count": step_count,
        "collision_count": engine_collision_count,      # engine predicate (N-C numerator)
        "nc_rate": engine_collision_count / float(step_count) if step_count else 0.0,
        "collision_event_count": collision_event_count,
        "dirty_success_flag": dirty_success,
        "proxy_collision_count": proxy_collision_count,  # displacement proxy (P2 object)
        "min_clearance_m": min(clearance_values_m) if clearance_values_m else UNAVAILABLE,
        "clearance_p10_m": float(np.percentile(clearance_values_m, 10)) if clearance_values_m else UNAVAILABLE,
        "clearance_p50_m": float(np.percentile(clearance_values_m, 50)) if clearance_values_m else UNAVAILABLE,
        "near_miss_count": near_miss_count,
        "time_in_risk_rate": time_in_risk_steps / float(len(clearance_values_m)) if clearance_values_m else UNAVAILABLE,
        "stuck_flag": int(max_consecutive_no_progress >= _STUCK_STREAK),
        "stuck_event_count": stuck_event_count,
        "guard_abort_count": UNAVAILABLE if counts is None else counts.abort_total,
        "guard_yield_count": UNAVAILABLE if counts is None else counts.yield_total,
        "tryout_escape_count": provider.tryout_escape_count() if tryout else UNAVAILABLE,
    })
