"""How precise is the guard's depth veto, and what does its frame memory add?

What: re-simulate episodes of an UNGUARDED landed arm from its logged actions (the
simulator is deterministic; same divergence gate as scripts/replay_geometry.py). On such an
arm every proposed forward step executed, so its engine collision flag is ground truth for
"this forward would have collided". At every step, the guard's projection (the rig the live
guard uses for this agent, as in vlnsafe/cli.py) is registered into world (x, z) by the
simulator pose, exactly as vlnsafe/guard.py does. At every executed forward step, the script
writes the forward-step clearance under a memory of M frames, for each M in --memory. A
veto at threshold t is then `clearance_M < t`, and the analysis reads off precision and
recall per M without touching any policy.

`--ranges` adds range-gated variants: older frames keep only the points they saw within R
metres (the current frame keeps all), column clear_m<M>_r<R>. The hypothesis under test:
far-range floor pixels misread as obstacles, kept after the current frame sees them as floor.

Writes <out>/guard_memory.csv.gz with one row per executed forward step: episode_id,
step_index, collided, displacement_m, clear_m<M>..., clear_m<M>_r<R>... Failures go to
<out>/guard_memory_failures.csv.

Run in the `vlnsafe-harness` env on a GPU with a working EGL context (not GPU 0):
  CUDA_VISIBLE_DEVICES=<i> python scripts/replay_guard_memory.py cma r2r block --limit 5
"""
import argparse
import csv
import gzip
import os
import sys
import time
from collections import deque

import numpy as np
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
    ap.add_argument("--out", help="default: data/scratch/<arm>/guard_memory_replay")
    ap.add_argument("--memory", default="1,2,4,8", help="memory lengths in frames")
    ap.add_argument("--ranges", default="", help="e.g. 1.0,1.5,2.0 (with the longest memory)")
    ap.add_argument("--limit", type=int, help="first N episodes after sharding")
    ap.add_argument("--every", type=int, default=1, help="every k-th episode in log order")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    a = ap.parse_args()
    memories = [int(m) for m in a.memory.split(",")]
    ranges = [float(r) for r in a.ranges.split(",") if r]

    arm = os.path.join(REPO, "data", "scratch", a.agent, a.benchmark, a.regime)
    steps_path = a.steps or os.path.join(arm, "steps.csv.gz")
    out = a.out or os.path.join(arm, "guard_memory_replay")
    os.makedirs(out, exist_ok=True)

    steps = pd.read_csv(
        steps_path, low_memory=False, dtype={"episode_id": str},
        usecols=["episode_id", "step_index", "proposed_action", "executed_action",
                 "distance_to_goal_m", "engine_collision_flag", "sim_displacement_m"])
    steps = steps.drop_duplicates(["episode_id", "step_index"])   # resume duplicates
    episode_ids = list(dict.fromkeys(steps["episode_id"]))[::a.every]
    episode_ids = episode_ids[a.shard_index::a.shards]
    if a.limit:
        episode_ids = episode_ids[:a.limit]
    steps = steps[steps["episode_id"].isin(set(episode_ids))]

    from replay_event import _build_env
    env, habitat_config = _build_env(a.agent, a.benchmark, a.regime, episode_ids)

    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from vlnsafe import config as cfg
    from vlnsafe.cli import _R2R_CLEARANCE_KEYS, _RXR_CLEARANCE_KEYS
    from vlnsafe.clearance_predictor import PrimitiveClearancePredictor
    from vlnsafe.geometry import CLEARANCE_DEPTH_SENSOR_KEYS
    from vlnsafe.guard import ego_to_world, world_to_ego
    from vlnsafe.simulator.pose import action_to_step, agent_pose

    # the rig the live guard projects (vlnsafe/cli.py): T2 workers -> the policy rig's
    # depth cameras; everything else -> the passive clearance ring
    if cfg.load_agent(a.agent).get("family") == "etpnav_worker":
        keys = _R2R_CLEARANCE_KEYS if a.benchmark == "r2r" else _RXR_CLEARANCE_KEYS
        predictor = PrimitiveClearancePredictor.from_habitat_config(
            habitat_config, sensor_keys=keys, sensor_name="DEPTH_SENSOR")
    else:
        predictor = PrimitiveClearancePredictor.from_habitat_config(
            habitat_config, sensor_keys=CLEARANCE_DEPTH_SENSOR_KEYS,
            sensor_name="CLEARANCE_DEPTH_0")
    act = {n: int(getattr(HabitatSimActions, n.upper())) for n in _NAMES}
    step_m, radius = predictor.forward_unit_m, predictor.agent_radius_m

    fails, done, n_rows = [], 0, 0
    start = time.time()
    with gzip.open(os.path.join(out, "guard_memory.csv.gz"), "wt") as gz:
        w = csv.writer(gz)
        w.writerow(["episode_id", "step_index", "collided", "displacement_m"]
                   + ["clear_m{}".format(m) for m in memories]
                   + ["clear_m{}_r{}".format(max(memories), r) for r in ranges])
        for eid in episode_ids:
            rows = steps[steps["episode_id"] == eid].sort_values("step_index")
            obs = env.reset()
            if str(env.current_episode.episode_id) != eid:
                fails.append((eid, "episode order: got {}".format(env.current_episode.episode_id)))
                continue
            memory = deque(maxlen=max(memories))
            block, bad = [], None
            for _, r in rows.iterrows():
                position, rotation = agent_pose(env)
                pts = np.asarray(predictor.points(obs), dtype=np.float64).reshape(-1, 2)
                memory.append((ego_to_world(pts, position, rotation), np.hypot(pts[:, 0], pts[:, 1])))
                clear = []
                if r["executed_action"] == "move_forward":
                    frames = list(memory)
                    cols = [np.concatenate([f[0] for f in frames[-m:]]) for m in memories]
                    for rng in ranges:
                        cols.append(np.concatenate([frames[-1][0]] + [
                            f[0][f[1] <= rng] for f in frames[-max(memories):-1]]))
                    for world in cols:
                        ego = world_to_ego(world, position, rotation)
                        clear.append("" if ego.shape[0] == 0 else round(
                            float(np.hypot(ego[:, 0] - step_m, ego[:, 1]).min()) - radius, 4))
                obs = env.step(action_to_step(act[r["executed_action"]]))
                hit = bool(env.sim.previous_step_collided)
                dtg = env.get_metrics().get("distance_to_goal")
                if hit != bool(r["engine_collision_flag"]) or (
                        dtg is not None and abs(float(dtg) - float(r["distance_to_goal_m"])) > TOL_M):
                    bad = "step {}: replay diverged from the log".format(int(r["step_index"]))
                    break
                if r["executed_action"] == "move_forward":
                    block.append([eid, int(r["step_index"]), int(hit), r["sim_displacement_m"]] + clear)
            if bad is None and not env.episode_over:
                bad = "log ends before the episode does"
            if bad:
                fails.append((eid, bad))
                continue
            w.writerows(block)
            done += 1
            n_rows += len(block)
    env.close()

    secs = time.time() - start
    with open(os.path.join(out, "guard_memory_failures.csv"), "w") as f:
        cw = csv.writer(f)
        cw.writerow(["episode_id", "reason"])
        cw.writerows(fails)
    print("replayed {} episodes ({} forward steps) in {:.0f}s = {:.1f} s/episode, {} failures".format(
        done, n_rows, secs, secs / max(done, 1), len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
