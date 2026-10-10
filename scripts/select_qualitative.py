#!/usr/bin/env python3
"""Pick the supplementary qualitative episodes by rule, never by eye.

Each case pairs an unguarded arm with its guarded arm on one episode. Among the episodes
that satisfy a case's condition, the pick is the one whose guarded episode length is
closest to the median guarded length of that set (ties: smallest episode id), so the
example is typical of its set rather than its most flattering member.

Writes data/scratch/qualitative/cases.json for scripts/render_qualitative.py.
Modern Python (pandas); never imported by the harness.
"""
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "scratch", "qualitative", "cases.json")


def paired(agent, bench, regime):
    d = os.path.join(REPO, "results", agent, bench)
    off = pd.read_csv(os.path.join(d, regime, "episodes.csv"), dtype={"episode_id": str})
    on = pd.read_csv(os.path.join(d, regime + "_guard-stall-mem1", "episodes.csv"),
                     dtype={"episode_id": str})
    return off.merge(on, on="episode_id", suffixes=("_off", "_on"))


def residual_escape_episodes(agent, bench, regime):
    """Guarded episodes with a collision on an escape (maneuver) forward."""
    s = pd.read_csv(os.path.join(REPO, "results", agent, bench, regime + "_guard-stall-mem1",
                                 "steps.csv.gz"),
                    usecols=["episode_id", "guard_action", "engine_collision_flag"],
                    dtype={"episode_id": str}, low_memory=False)
    hit = s[(s.guard_action == "maneuver") & (s.engine_collision_flag == 1)]
    return set(hit.episode_id)


def pick(df):
    med = df.step_count_on.median()
    df = df.assign(dist=(df.step_count_on - med).abs(), eid=df.episode_id.astype(int))
    return str(df.sort_values(["dist", "eid"]).iloc[0].episode_id)


def main():
    cases = []
    spec = [
        ("rescue_topological", "etpr1", "r2r", "block",
         "stuck failure without the guard, success with it",
         lambda p: p[(p.stuck_flag_off == 1) & (p.success_off == 0) & (p.success_on == 1)]),
        ("rescue_vla", "omninav", "r2r", "block",
         "stuck failure without the guard, success with it",
         lambda p: p[(p.stuck_flag_off == 1) & (p.success_off == 0) & (p.success_on == 1)]),
        ("clean_under_slide", "etpr1", "r2r", "slide",
         "dirty success without the guard, clean success with it",
         lambda p: p[(p.dirty_success_flag_off == 1) & (p.success_on == 1)
                     & (p.collision_count_on == 0)]),
        ("residual_escape_contact", "etpnav", "r2r", "block",
         "guarded success with a collision on an escape forward",
         None),
        ("guard_failure", "etpr1", "r2r", "block",
         "success without the guard, failure with it",
         lambda p: p[(p.success_off == 1) & (p.success_on == 0)]),
    ]
    for name, agent, bench, regime, desc, cond in spec:
        p = paired(agent, bench, regime)
        if cond is None:
            esc = residual_escape_episodes(agent, bench, regime)
            sel = p[p.episode_id.isin(esc) & (p.success_on == 1) & (p.success_off == 0)]
        else:
            sel = cond(p)
        eid = pick(sel)
        row = p[p.episode_id == eid].iloc[0]
        cases.append({"case": name, "agent": agent, "benchmark": bench, "regime": regime,
                      "episode_id": eid, "condition": desc, "n_eligible": int(len(sel)),
                      "off": {"success": int(row.success_off), "steps": int(row.step_count_off),
                              "collisions": int(row.collision_count_off)},
                      "on": {"success": int(row.success_on), "steps": int(row.step_count_on),
                             "collisions": int(row.collision_count_on)}})
        print(cases[-1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(cases, f, indent=1)


if __name__ == "__main__":
    main()
