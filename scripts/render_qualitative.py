"""Replay the supplementary qualitative episodes and dump what the figure needs.

For every case in data/scratch/qualitative/cases.json (scripts/select_qualitative.py),
rebuild the arm's habitat env exactly as `vlnsafe run` does (scripts/replay_event.py's
_build_env), and replay the same episode twice from the logged executed_action column:
once from the unguarded arm, once from the guarded arm. No agent and no model -- the
simulator is deterministic. Every replayed step's engine collision flag and
distance-to-goal must match the log, or the case is reported as failed, never drawn.

Writes, per case, <out>/<case>.npz (top-down navmesh map and bounds, both paths, per-step
collision flags and guard actions, goal and start) and a few egocentric RGB frames
(<out>/<case>_<arm>_<step>.png) at the guarded arm's first veto and the unguarded arm's
first collision. scripts/plot_qualitative.py draws the figure from these files.

Run in the `vlnsafe-harness` env (py3.6, habitat-sim 0.1.7). It renders, so it needs a GPU
with a working EGL context (not GPU 0): CUDA_VISIBLE_DEVICES=<i> python scripts/render_qualitative.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from replay_event import TOL_M, _build_env, _rgb_key  # noqa: E402

COLS = ["episode_id", "step_index", "executed_action", "distance_to_goal_m",
        "engine_collision_flag"]


def _steps(agent, bench, arm, eid):
    p = os.path.join(REPO, "results", agent, bench, arm, "steps.csv.gz")
    cols = COLS + (["guard_action"] if "_guard-" in arm else [])
    s = pd.read_csv(p, usecols=cols, dtype={"episode_id": str}, low_memory=False)
    s = s[s.episode_id == eid].drop_duplicates("step_index").sort_values("step_index")
    if "guard_action" not in s:
        s["guard_action"] = "none"
    return s.reset_index(drop=True)


def _replay(env, act, rows, eid, frame_steps, out_prefix):
    import imageio
    obs = env.reset()
    key = _rgb_key(obs)
    if str(env.current_episode.episode_id) != eid:
        raise RuntimeError("episode order: got {}".format(env.current_episode.episode_id))
    path = [np.array(env.sim.get_agent_state().position, dtype=float)]
    hits = []
    for i in range(len(rows)):
        r = rows.iloc[i]
        obs = env.step({"action": act[r["executed_action"]]})
        path.append(np.array(env.sim.get_agent_state().position, dtype=float))
        hit = bool(env.sim.previous_step_collided)
        dtg = env.get_metrics().get("distance_to_goal")
        if hit != bool(r["engine_collision_flag"]) or (
                dtg is not None and abs(float(dtg) - float(r["distance_to_goal_m"])) > TOL_M):
            raise RuntimeError("step {}: replay diverged from the log".format(i))
        hits.append(hit)
        if key and i in frame_steps:
            imageio.imwrite("{}_{}.png".format(out_prefix, i),
                            np.array(obs[key][..., :3], dtype=np.uint8))
        if env.episode_over:
            break
    return np.array(path), np.array(hits, dtype=bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=os.path.join(REPO, "data", "scratch", "qualitative",
                                                    "cases.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "scratch", "qualitative"))
    a = ap.parse_args()
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.utils.visualizations import maps

    act = {"stop": int(HabitatSimActions.STOP),
           "move_forward": int(HabitatSimActions.MOVE_FORWARD),
           "turn_left": int(HabitatSimActions.TURN_LEFT),
           "turn_right": int(HabitatSimActions.TURN_RIGHT)}
    with open(a.cases) as f:
        cases = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    failed = []
    for c in cases:
        agent, bench, regime, eid = c["agent"], c["benchmark"], c["regime"], c["episode_id"]
        off = _steps(agent, bench, regime, eid)
        on = _steps(agent, bench, regime + "_guard-stall-mem1", eid)
        vetoes = on.index[on.guard_action.isin(["abort", "stall"])].tolist()
        first_hit_off = off.index[off.engine_collision_flag == 1].tolist()
        frames_on = set()
        if vetoes:  # the view the policy proposed forward from, and the view after escape
            v, end = vetoes[0], vetoes[0]
            while end + 1 < len(on) and on.guard_action[end + 1] == "maneuver":
                end += 1
            frames_on = {max(v - 1, 0), end}
        frames_off = set(first_hit_off[:1])
        here = os.getcwd()
        env, _ = _build_env(agent, bench, regime, [eid, eid])
        try:
            p_off, h_off = _replay(env, act, off, eid, frames_off,
                                   os.path.join(a.out, c["case"] + "_off"))
            p_on, h_on = _replay(env, act, on, eid, frames_on,
                                 os.path.join(a.out, c["case"] + "_on"))
            height = float(p_on[0][1])
            m = maps.get_topdown_map(env.sim.pathfinder, height, 1024, False)
            b = env.sim.pathfinder.get_bounds()
            goal = np.array(env.current_episode.goals[0].position, dtype=float)
        except RuntimeError as e:
            failed.append((c["case"], str(e)))
            print("[{}] FAILED: {}".format(c["case"], e), flush=True)
            continue
        finally:
            env.close()
            os.chdir(here)
        np.savez_compressed(
            os.path.join(a.out, c["case"] + ".npz"), topdown=np.asarray(m),
            bounds=np.array([[b[0][0], b[0][2]], [b[1][0], b[1][2]]], dtype=float),
            path_off=p_off, hits_off=h_off, path_on=p_on, hits_on=h_on,
            guard_on=np.array(on.guard_action.tolist()[:len(h_on)]), goal=goal,
            frames_on=np.array(sorted(frames_on)), frames_off=np.array(sorted(frames_off)))
        print("[{}] ok: off {} steps ({} hits), on {} steps ({} hits)".format(
            c["case"], len(h_off), int(h_off.sum()), len(h_on), int(h_on.sum())), flush=True)
    print("{} cases, {} failed".format(len(cases), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
