"""Re-simulate logged contact events from their executed actions (no model needed).

What: for each event in an events-key CSV (one sampled contact onset per row), rebuild the arm's habitat env
exactly as `vlnsafe run` does, replay the episode from its logged executed_action column
(no agent, no GPU model -- the simulator is deterministic), and for steps [t-8, t+8] around
the contact onset write
  <out>/<event_id>/clip.gif     the agent's RGB view over the window (contact steps get a
                                red border)
  <out>/<event_id>/trace.png    top-down map with the whole path, the window in colour,
                                contact steps marked, start/goal marked, 1 m scale bar
and fill net_disp_m (straight-line distance between the agent at t-8 and t+8, the only
feature that needs positions) into a coding-sheet CSV beside it. The replay helpers are
also imported by render_qualitative.py, replay_geometry.py and replay_guard_memory.py.

Determinism check: every replayed step's distance_to_goal_m and engine collision flag must
match the logged row; the first mismatch aborts that event and is listed in
<out>/replay_failures.csv, never silently coded from a diverged replay.

Neither the clip nor the trace shows agent or outcome; the file names carry only the
opaque event_id (blinding).

Run in the `vlnsafe-harness` env (py3.6, habitat-sim 0.1.7). It renders, so it needs a GPU
with a working EGL context (not GPU 0): CUDA_VISIBLE_DEVICES=<i> python scripts/replay_event.py
Scratch output goes under data/scratch/taxonomy by default (videos are never committed).
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

WINDOW = 8
TOL_M = 1e-3


def _rgb_key(obs):
    if "rgb" in obs:
        return "rgb"
    keys = sorted(k for k in obs if k.startswith("rgb"))
    return keys[0] if keys else None


def _build_env(agent, bench_id, regime, episode_ids):
    """The env of `vlnsafe run --agent A --benchmark B --regime R`, minus the provider,
    restricted to episode_ids. Mirrors vlnsafe/cli.py::cmd_run."""
    from vlnsafe import config as cfg

    run_cfg = cfg.load_run(agent, bench_id, regime)
    bench, ag = run_cfg["benchmark"], run_cfg["agent"]
    seed = run_cfg["defaults"]["seed"]
    split = bench["split"]
    vlnce_root = os.path.join(cfg.repo_root(), "external", "VLN-CE")
    sys.path.insert(0, vlnce_root)
    os.chdir(vlnce_root)

    from vlnsafe.simulator import environment as envmod

    ds_key = cfg.agent_dataset_key(ag, bench_id)
    instr_root = cfg.benchmark_dataset_root(bench, ds_key)
    instr_type = cfg.benchmark_dataset_type(bench, ds_key)
    if bench_id == "r2r":
        hc = envmod.build_r2r_config(
            bench["habitat_base_config"], split, bench["dataset"]["official"],
            bench["dataset"]["scenes"], instr_root, seed, regime,
            ag["sensor_rig"], ag.get("rig"), dataset_type=instr_type)
    else:
        sel = bench.get("select", {})
        seed = sel.get("seed", seed)
        hc = envmod.build_rxr_config(
            bench["habitat_base_config"], split, instr_root, bench["dataset"]["scenes"],
            seed, regime, ag["sensor_rig"], ag.get("rig"))
    env = envmod.build_env(hc)
    wanted = [str(e) for e in episode_ids]
    envmod.filter_env_episodes_ordered(env, [{"episode_id": e} for e in wanted])
    return env, hc


def _topdown(env, path, lo, hi, contact_idx, goal, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xz = np.array([[p[0], p[2]] for p in path])
    fig, ax = plt.subplots(figsize=(6, 6))
    try:  # the navmesh, so walls are visible; falls back to a bare trace
        from habitat.utils.visualizations import maps
        h = path[hi][1]
        m = maps.get_topdown_map(env.sim.pathfinder, h, 1024, False)
        b0 = env.sim.pathfinder.get_bounds()
        ax.imshow(m == 0, cmap="Greys", alpha=0.35, origin="upper",
                  extent=[b0[0][0], b0[1][0], b0[1][2], b0[0][2]])
    except Exception:
        pass
    ax.plot(xz[:, 0], xz[:, 1], "-", color="0.7", lw=1)
    ax.plot(xz[lo:hi + 1, 0], xz[lo:hi + 1, 1], "-o", color="tab:blue", ms=3, lw=1.5)
    if contact_idx:
        ax.plot(xz[contact_idx, 0], xz[contact_idx, 1], "x", color="red", ms=8, mew=2)
    ax.plot(*xz[0], "s", color="green", ms=7)
    ax.plot(goal[0], goal[2], "*", color="gold", ms=14, mec="k")
    cx, cy = xz[lo:hi + 1, 0].mean(), xz[lo:hi + 1, 1].mean()
    ax.set_xlim(cx - 3.5, cx + 3.5)
    ax.set_ylim(cy + 3.5, cy - 3.5)
    ax.set_aspect("equal")
    ax.plot([cx - 3.3, cx - 2.3], [cy + 3.2, cy + 3.2], "k-", lw=3)
    ax.text(cx - 2.8, cy + 3.0, "1 m", ha="center", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _replay_arm(agent, bench, regime, events, steps, out_root, fails):
    import imageio
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    act = {"stop": int(HabitatSimActions.STOP),
           "move_forward": int(HabitatSimActions.MOVE_FORWARD),
           "turn_left": int(HabitatSimActions.TURN_LEFT),
           "turn_right": int(HabitatSimActions.TURN_RIGHT)}
    env, _ = _build_env(agent, bench, regime, list(events["episode_id"]))
    net = {}
    try:
        for _, ev in events.iterrows():
            eid, t, evid = str(ev["episode_id"]), int(ev["step"]), ev["event_id"]
            rows = steps[steps["episode_id"] == eid].sort_values("step_index")
            lo_s, hi_s = max(0, t - WINDOW), min(len(rows) - 1, t + WINDOW)
            obs = env.reset()
            if str(env.current_episode.episode_id) != eid:
                fails.append((evid, "episode order: got {}".format(
                    env.current_episode.episode_id)))
                continue
            key = _rgb_key(obs)
            goal = env.current_episode.goals[0].position
            path = [np.array(env.sim.get_agent_state().position, dtype=float)]
            frames, contact_idx, bad = {}, [], None
            for i in range(hi_s + 1):
                r = rows.iloc[i]
                obs = env.step({"action": act[r["executed_action"]]})
                path.append(np.array(env.sim.get_agent_state().position, dtype=float))
                hit = bool(env.sim.previous_step_collided)
                dtg = env.get_metrics().get("distance_to_goal")
                if hit != bool(r["engine_collision_flag"]) or (
                        dtg is not None and abs(float(dtg) - float(r["distance_to_goal_m"])) > TOL_M):
                    bad = "step {}: replay diverged from the log".format(i)
                    break
                if lo_s <= i <= hi_s:
                    if hit:
                        contact_idx.append(i + 1)  # path[j] is the pose after step j-1
                    if key:
                        f = np.array(obs[key][..., :3], dtype=np.uint8)
                        if hit:
                            f[:6], f[-6:], f[:, :6], f[:, -6:] = 255, 255, 255, 255
                            f[:6, :, 1:], f[-6:, :, 1:] = 0, 0
                            f[:, :6, 1:], f[:, -6:, 1:] = 0, 0
                        frames[i] = f
                if env.episode_over and i < hi_s:
                    bad = "episode ended early at step {}".format(i)
                    break
            if bad:
                fails.append((evid, bad))
                continue
            d = os.path.join(out_root, evid)
            os.makedirs(d, exist_ok=True)
            if frames:
                imageio.mimsave(os.path.join(d, "clip.gif"),
                                [frames[i] for i in sorted(frames)], duration=0.5)
            _topdown(env, path, lo_s, hi_s + 1, contact_idx, goal,
                     os.path.join(d, "trace.png"))
            net[evid] = round(float(np.linalg.norm(path[hi_s + 1] - path[lo_s + 1])), 3)
    finally:
        env.close()
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--taxonomy", default=os.path.join(REPO, "results", "taxonomy"))
    ap.add_argument("--results", default=os.path.join(REPO, "results"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "scratch", "taxonomy"))
    ap.add_argument("--only", nargs="*", help="event ids (default: all)")
    ap.add_argument("--limit", type=int, help="stop after this many events (smoke test)")
    a = ap.parse_args()

    key = pd.read_csv(os.path.join(a.taxonomy, "events_key.csv"), dtype={"episode_id": str})
    if a.only:
        key = key[key["event_id"].isin(a.only)]
    if a.limit:
        key = key.head(a.limit)
    os.makedirs(a.out, exist_ok=True)
    fails, net = [], {}
    for (agent, bench, regime), ev in key.groupby(["agent", "benchmark", "regime"]):
        d = os.path.join(a.results, agent, bench, regime)
        steps = pd.read_csv(os.path.join(d, "steps.csv.gz"), low_memory=False,
                            usecols=["episode_id", "step_index", "executed_action",
                                     "distance_to_goal_m", "engine_collision_flag"],
                            dtype={"episode_id": str})
        print("[{}/{}/{}] replaying {} events".format(agent, bench, regime, len(ev)), flush=True)
        net.update(_replay_arm(agent, bench, regime, ev, steps, a.out, fails))

    for name in ("coding_sheet.csv", "coding_sheet_r2.csv"):
        p = os.path.join(a.taxonomy, name)
        sheet = pd.read_csv(p, dtype={"net_disp_m": object}).fillna("")
        for i, evid in enumerate(sheet["event_id"]):
            if evid in net:
                sheet.at[i, "net_disp_m"] = net[evid]
        sheet.to_csv(p, index=False)
    with open(os.path.join(a.out, "replay_failures.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "reason"])
        w.writerows(fails)
    print("replayed {} events, {} failures".format(len(net), len(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
