#!/usr/bin/env python3
"""The method result: a guarded arm against the
unguarded arm it was cut from, paired on episode_id.

For every landed results/<agent>/<benchmark>/<regime>_guard-<name>/ with its base arm
results/<agent>/<benchmark>/<regime>/, reports:

  - task and safety means, guard off vs on, with the paired delta and a 95% bootstrap CI
    over episodes: SR, SPL, nDTW, NE, N-C rate, collision steps and events per episode,
    dirty success, stuck flag and stuck events, steps per episode. "Report both or the
    result is dishonest" (section 6): a guard that trades collisions for stuck episodes
    shows up here.
  - intervention: aborts and yields per episode, share of episodes with any abort, aborts
    per 100 executed steps, stall vetoes per episode (guard `stall`), the share of steps
    the guard owned (abort + stall + maneuver).
  - the N21 budget check: share of episodes whose decision count reached the agent's
    cap (ETP-R1: IL.max_traj_len = 15), off vs on. The decided rule: >= 3 pp more when
    guarded -> run the exempting fallback arm. For an arm run with
    --exempt-guard-redecisions (manifest guard.exempt_guard_redecisions) the cap is
    extended per guard-forced re-decision, so "over cap" is expected there;
    episodes_over_allowed_cap_on is the check that must be 0 for both.
  - residual engine collisions by guard_action (pass = the guard never flagged it).
  - success flips: episodes the guard turned from failure to success (gained) and back (lost).

Also, for every agent-own baseline arm results/<agent>/<benchmark>/<regime>_tryout/
, the same paired task/safety block against the unguarded arm
(`vs_off`), the method guard arm against it (`guard_vs_baseline`) and, when
`<regime>_tryout_guard-<method>` exists, that arm against the guard arm
(`combo_vs_guard`; against the baseline it is an ordinary guard entry), written to
results/summary/baseline_effect.json. `clean_success` (success with no engine collision)
is in every comparison: Tryout cannot add any, since an episode without a collision runs
identically with it on or off.

Usage: analysis/guard_effect.py [--out results/summary/guard_effect.json]
       analysis/guard_effect.py --pair OFF_DIR ON_DIR   (print one pair, e.g. a smoke run)
Modern Python (pandas, numpy); never imported by the harness.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DECISION_CAP = {"etpnav": 15, "etpr1": 15}   # IL.max_traj_len in each worker's r2r config
BUDGET_RULE_PP = 3.0
METRICS = ("success", "spl", "ndtw", "navigation_error_m", "nc_rate", "collision_count",
           "collision_event_count", "dirty_success_flag", "stuck_flag", "stuck_event_count",
           "step_count", "clean_success")
BASELINE_ARMS = ("block_tryout",)
METHOD = "stall-mem1"   # the method guard config (analysis/make_tables.py METHOD)
N_BOOT = 2000
SEED = 20260704


def _read_arm(arm_dir, episode_ids=None):
    ep = pd.read_csv(os.path.join(arm_dir, "episodes.csv"))
    ep["episode_id"] = ep["episode_id"].astype(str)
    ep["clean_success"] = ((pd.to_numeric(ep["success"], errors="coerce") == 1)
                           & (pd.to_numeric(ep["collision_count"], errors="coerce") == 0)
                           ).astype(float)
    steps_path = os.path.join(arm_dir, "steps.csv.gz")
    if not os.path.exists(steps_path):
        steps_path = os.path.join(arm_dir, "steps.csv")
    cols = ["episode_id", "is_decision", "engine_collision_flag"]
    header = pd.read_csv(steps_path, nrows=0).columns
    cols += [c for c in ("guard_action",) if c in header]
    st = pd.read_csv(steps_path, usecols=cols, dtype={"episode_id": str})
    if episode_ids is not None:
        ep = ep[ep.episode_id.isin(episode_ids)]
        st = st[st.episode_id.isin(episode_ids)]
    return ep.drop_duplicates("episode_id").set_index("episode_id"), st


def _boot_ci(diff, rng):
    idx = rng.randint(0, len(diff), size=(N_BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def compare(off_dir, on_dir, agent=None, guard=True):
    """One paired comparison over the episodes both arms contain. guard=False stops after
    the task/safety metrics and the success flips (an `on` arm without the guard)."""
    ep_on, st_on = _read_arm(on_dir)
    ep_off, st_off = _read_arm(off_dir, set(ep_on.index))
    shared = ep_on.index.intersection(ep_off.index)
    ep_on, ep_off = ep_on.loc[shared], ep_off.loc[shared]
    st_on = st_on[st_on.episode_id.isin(shared)]
    rng = np.random.RandomState(SEED)

    out = {"off_dir": os.path.relpath(off_dir, REPO), "on_dir": os.path.relpath(on_dir, REPO),
           "episodes": int(len(shared)), "metrics": {}}
    for m in METRICS:
        a = pd.to_numeric(ep_off[m], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(ep_on[m], errors="coerce").to_numpy(dtype=float)
        ok = ~(np.isnan(a) | np.isnan(b))
        d = b[ok] - a[ok]
        out["metrics"][m] = {"off": float(a[ok].mean()), "on": float(b[ok].mean()),
                             "delta": float(d.mean()), "ci95": _boot_ci(d, rng),
                             "n": int(ok.sum())}

    s_off = pd.to_numeric(ep_off["success"], errors="coerce").fillna(0).astype(int)
    s_on = pd.to_numeric(ep_on["success"], errors="coerce").fillna(0).astype(int)
    out["success_flips"] = {"gained": int(((s_off == 0) & (s_on == 1)).sum()),
                            "lost": int(((s_off == 1) & (s_on == 0)).sum())}
    if not guard:
        return out

    aborts = pd.to_numeric(ep_on["guard_abort_count"], errors="coerce")
    yields = pd.to_numeric(ep_on["guard_yield_count"], errors="coerce")
    actions = st_on["guard_action"].value_counts()
    n_steps = float(len(st_on))
    stalls = st_on.assign(s=st_on.guard_action == "stall").groupby("episode_id").s.sum()
    out["intervention"] = {
        "aborts_per_episode": float(aborts.mean()),
        "yields_per_episode": float(yields.mean()),
        "episodes_with_abort": float((aborts > 0).mean()),
        "aborts_per_100_steps": float(100.0 * actions.get("abort", 0) / n_steps),
        "stall_vetoes_per_episode": float(stalls.reindex(ep_on.index, fill_value=0).mean()),
        "guard_owned_step_share": float(
            (actions.get("abort", 0) + actions.get("stall", 0) + actions.get("maneuver", 0))
            / n_steps),
        "step_actions": {k: int(v) for k, v in actions.items()},
    }
    out["residual_collisions_by_guard_action"] = {
        k: int(v) for k, v in st_on.groupby("guard_action").engine_collision_flag.sum().items()}

    manifest_path = os.path.join(on_dir, "manifest.json")
    guard_meta = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            guard_meta = json.load(f).get("guard") or {}
    exempt = bool(guard_meta.get("exempt_guard_redecisions"))
    out["exempt_guard_redecisions"] = exempt

    cap = DECISION_CAP.get(agent)
    if cap is not None:
        dec_on = st_on.groupby("episode_id").is_decision.sum()
        # every veto (depth abort or stall) installs a manoeuvre, hence one exempt re-decision
        aborts_by_ep = st_on.assign(a=st_on.guard_action.isin(["abort", "stall"])).groupby(
            "episode_id").a.sum()
        st_off_shared = st_off[st_off.episode_id.isin(shared)]
        hit_off = float((st_off_shared.groupby("episode_id").is_decision.sum() >= cap).mean())
        hit_on = float((st_on.groupby("episode_id").is_decision.sum() >= cap).mean())
        delta_pp = 100.0 * (hit_on - hit_off)
        out["decision_budget"] = {
            "cap": cap, "cap_hit_share_off": hit_off, "cap_hit_share_on": hit_on,
            "delta_pp": delta_pp, "rule_pp": BUDGET_RULE_PP,
            "fallback_arm_needed": bool(delta_pp >= BUDGET_RULE_PP),
            # must be 0. Budget counted (M1): a decision past the cap means the worker
            # re-decided after a stop (N10's first launch). Budget exempt (M1b): the
            # policy may exceed the cap by one per guard-forced re-decision, of which there
            # are at most as many as vetoes.
            "episodes_over_allowed_cap_on": int(
                (dec_on > cap + (aborts_by_ep if exempt else 0)).sum()),
            "episodes_over_cap_on": int((dec_on > cap).sum()),
            "decisions_per_episode_off": float(st_off_shared.groupby("episode_id").is_decision.sum().mean()),
            "decisions_per_episode_on": float(st_on.groupby("episode_id").is_decision.sum().mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results", "summary", "guard_effect.json"))
    ap.add_argument("--pair", nargs=2, metavar=("OFF_DIR", "ON_DIR"))
    ap.add_argument("--agent", default=None, help="with --pair: agent id, for the decision cap")
    args = ap.parse_args()

    if args.pair:
        print(json.dumps(compare(args.pair[0], args.pair[1], args.agent), indent=2))
        return 0

    arms = {}
    for on_dir in sorted(glob.glob(os.path.join(REPO, "results", "*", "*", "*_guard-*"))):
        if not os.path.exists(os.path.join(on_dir, "episodes.csv")):
            continue
        agent, bench, arm = on_dir.split(os.sep)[-3:]
        regime = arm.split("_guard-")[0]
        off_dir = os.path.join(REPO, "results", agent, bench, regime)
        arms["/".join((agent, bench, arm))] = compare(off_dir, on_dir, agent)
    with open(args.out, "w") as f:
        json.dump(arms, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote {} ({} guarded arms)".format(args.out, len(arms)))

    baselines = {}
    for name in BASELINE_ARMS:
        for base_dir in sorted(glob.glob(os.path.join(REPO, "results", "*", "*", name))):
            if not os.path.exists(os.path.join(base_dir, "episodes.csv")):
                continue
            agent, bench, _ = base_dir.split(os.sep)[-3:]
            regime = name.split("_")[0]
            entry = {"vs_off": compare(os.path.join(REPO, "results", agent, bench, regime),
                                       base_dir, agent, guard=False)}
            guard_dir = os.path.join(REPO, "results", agent, bench,
                                     "{}_guard-{}".format(regime, METHOD))
            if os.path.exists(os.path.join(guard_dir, "episodes.csv")):
                entry["guard_vs_baseline"] = compare(base_dir, guard_dir, agent, guard=False)
                entry["guard_vs_baseline"]["guard"] = METHOD
                # The baseline with the method guard's veto inside it, against the guard
                combo_dir = "{}_guard-{}".format(base_dir, METHOD)
                if os.path.exists(os.path.join(combo_dir, "episodes.csv")):
                    entry["combo_vs_guard"] = compare(guard_dir, combo_dir, agent, guard=False)
            baselines["/".join((agent, bench, name))] = entry
    base_out = os.path.join(os.path.dirname(args.out), "baseline_effect.json")
    with open(base_out, "w") as f:
        json.dump(baselines, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote {} ({} baseline arms)".format(base_out, len(baselines)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
