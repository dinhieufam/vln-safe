"""Recover the external-geometry column for an arm that was run without it.

What: re-simulate each episode of an existing arm from its logged actions (no agent, no
model -- the simulator is deterministic) and, at every step, call the same
PrimitiveClearancePredictor `vlnsafe run` now uses on the depth frame the agent acted on.
Writes a sidecar <out>/geometry.csv.gz with (episode_id, step_index, predicted_clearance_m)
that lines up 1:1 with the arm's steps.csv.gz rows. Equivalent to re-running the arm with
the predictor on, at simulator cost only -- StreamVLN's 81 s/episode is model inference.

Determinism gate, same as scripts/replay_event.py: every replayed step's distance_to_goal_m
and engine collision flag must match the logged row. An episode that diverges is dropped
and listed in <out>/geometry_failures.csv, never written half-computed.

Run in the `vlnsafe-harness` env; it renders, so it needs a GPU with a working EGL context
(not GPU 0):
  CUDA_VISIBLE_DEVICES=<i> python scripts/replay_geometry.py cma r2r block --limit 5
"""
import argparse
import csv
import gzip
import os
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

TOL_M = 1e-3
_NAMES = ("stop", "move_forward", "turn_left", "turn_right")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("agent")
    ap.add_argument("benchmark", choices=["r2r", "rxr"])
    ap.add_argument("regime", choices=["block", "slide"])
    ap.add_argument("--steps", help="steps.csv.gz of the arm (default: data/scratch/<arm>)")
    ap.add_argument("--out", help="default: data/scratch/<arm>/geometry_replay")
    ap.add_argument("--limit", type=int, help="first N episodes in log order (smoke/timing)")
    ap.add_argument("--only", nargs="*", help="these episode ids only (e.g. a known stuck one)")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    a = ap.parse_args()

    arm = os.path.join(REPO, "data", "scratch", a.agent, a.benchmark, a.regime)
    steps_path = a.steps or os.path.join(arm, "steps.csv.gz")
    out = a.out or os.path.join(arm, "geometry_replay")
    os.makedirs(out, exist_ok=True)

    steps = pd.read_csv(
        steps_path, low_memory=False, dtype={"episode_id": str},
        usecols=["episode_id", "step_index", "proposed_action", "executed_action",
                 "distance_to_goal_m", "engine_collision_flag"])
    episode_ids = list(dict.fromkeys(steps["episode_id"]))   # log order
    episode_ids = episode_ids[a.shard_index::a.shards]
    if a.only:
        episode_ids = [e for e in episode_ids if e in set(a.only)]
    if a.limit:
        episode_ids = episode_ids[:a.limit]
    steps = steps[steps["episode_id"].isin(set(episode_ids))]

    from replay_event import _build_env   # env of `vlnsafe run`, restricted to the episodes
    env, habitat_config = _build_env(a.agent, a.benchmark, a.regime, episode_ids)

    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from vlnsafe.clearance_predictor import PrimitiveClearancePredictor
    from vlnsafe.geometry import CLEARANCE_DEPTH_SENSOR_KEYS
    from vlnsafe.records.schema import UNAVAILABLE
    from vlnsafe.simulator.pose import action_to_step

    act = {n: int(getattr(HabitatSimActions, n.upper())) for n in _NAMES}
    predictor = PrimitiveClearancePredictor.from_habitat_config(
        habitat_config, sensor_keys=CLEARANCE_DEPTH_SENSOR_KEYS,   # same as cli.py
        sensor_name="CLEARANCE_DEPTH_0")

    fails, done, n_steps = [], 0, 0
    start = time.time()
    with gzip.open(os.path.join(out, "geometry.csv.gz"), "wt") as gz:
        w = csv.writer(gz)
        w.writerow(["episode_id", "step_index", "predicted_clearance_m"])
        for eid in episode_ids:
            rows = steps[steps["episode_id"] == eid].sort_values("step_index")
            obs = env.reset()
            if str(env.current_episode.episode_id) != eid:
                fails.append((eid, "episode order: got {}".format(env.current_episode.episode_id)))
                continue
            block, bad = [], None
            for _, r in rows.iterrows():
                pred = predictor(obs, act[r["proposed_action"]])
                obs = env.step(action_to_step(act[r["executed_action"]]))
                hit = bool(env.sim.previous_step_collided)
                dtg = env.get_metrics().get("distance_to_goal")
                if hit != bool(r["engine_collision_flag"]) or (
                        dtg is not None and abs(float(dtg) - float(r["distance_to_goal_m"])) > TOL_M):
                    bad = "step {}: replay diverged from the log".format(int(r["step_index"]))
                    break
                block.append((eid, int(r["step_index"]), UNAVAILABLE if pred is None else pred))
            if bad is None and not env.episode_over:
                bad = "log ends before the episode does"
            if bad:
                fails.append((eid, bad))
                continue
            w.writerows(block)
            done += 1
            n_steps += len(block)
    env.close()

    secs = time.time() - start
    with open(os.path.join(out, "geometry_failures.csv"), "w") as f:
        cw = csv.writer(f)
        cw.writerow(["episode_id", "reason"])
        cw.writerows(fails)
    print("replayed {} episodes ({} steps) in {:.0f}s = {:.1f} s/episode, {} failures".format(
        done, n_steps, secs, secs / max(done, 1), len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
