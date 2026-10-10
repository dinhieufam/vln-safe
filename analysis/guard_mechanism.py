#!/usr/bin/env python3
"""Why the guard changes what it changes. Paired on episode_id,
every landed results/<agent>/<benchmark>/<regime>_guard-<name>/ against its unguarded arm.

  a. `sr_gain_by_unguarded_state`: each episode is put in one class by what the unguarded
     arm did in it -- `stuck` (a proxy stuck event), else `timeout` (hit the step limit),
     else `budget` (ETP family: used the whole decision budget), else `other`. Per class:
     episodes, SR off/on, flips gained/lost, and its share of the net SR gain. Plus the
     paired SR change on the episodes that were *not* stuck without the guard: the part of
     the gain that is not un-sticking.
  b. `residual_collisions`: the guarded arm's remaining collision steps by the primitive
     that caused them -- an unflagged policy forward (`pass`), a flagged forward the caps
     let through (`yield`), an escape forward (`maneuver`) -- as counts and shares.
     `fragmentation`: an event (maximal run of collision steps) that starts within
     FRAGMENT_GAP primitives of the previous one, with a guard-owned step in between, is a
     fragment of one contact the guard interrupted. Events per episode with fragments
     merged back is the event count comparable to the unguarded arm's. Also event length
     and contact per metre travelled.
  c. `length`: steps per episode, step-limit timeouts, off vs on (decision budgets are in
     guard_effect.json).
  e. `immobile`: a stuck measure the stall veto cannot satisfy by construction -- some
     window of IMMOBILE_WINDOW consecutive primitives moves the agent less than
     IMMOBILE_M in total (turning in place counts as immobile). The proxy `stuck_flag`
     (five consecutive near-zero forwards) is exactly what the stall veto forbids.
     Also written for every unguarded arm, as context.

Usage: analysis/guard_mechanism.py [--out results/summary/guard_mechanism.json]
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

MAX_STEPS = 500               # configs/defaults.yaml episode step limit
DECISION_CAP = {"etpnav": 15, "etpr1": 15}   # as in analysis/guard_effect.py
FRAGMENT_GAP = 12             # primitives: longer than one escape (<= 6 turns + 1 forward)
IMMOBILE_WINDOW = 40          # primitives
IMMOBILE_M = 0.5              # metres moved in that window (two forward steps)
GUARD_OWNED = ("abort", "stall", "maneuver", "stop")
N_BOOT = 2000
SEED = 20260927
STEP_COLS = ["episode_id", "step_index", "is_decision", "executed_action",
             "sim_displacement_m", "engine_collision_flag"]


def _read(arm_dir):
    ep = pd.read_csv(os.path.join(arm_dir, "episodes.csv"), dtype={"episode_id": str})
    ep = ep.drop_duplicates("episode_id", keep="last").set_index("episode_id")
    path = os.path.join(arm_dir, "steps.csv.gz")
    header = pd.read_csv(path, nrows=0).columns
    cols = STEP_COLS + [c for c in ("guard_action",) if c in header]
    st = pd.read_csv(path, usecols=cols, dtype={"episode_id": str})
    st = st.drop_duplicates(["episode_id", "step_index"], keep="last")
    st = st.sort_values(["episode_id", "step_index"]).reset_index(drop=True)
    for c in ("is_decision", "sim_displacement_m", "engine_collision_flag"):
        st[c] = pd.to_numeric(st[c], errors="coerce").fillna(0)
    return ep, st


def _num(ep, col):
    return pd.to_numeric(ep[col], errors="coerce").fillna(0)


def _paired(a, b, rng):
    d = np.asarray(b, float) - np.asarray(a, float)
    if len(d) == 0:
        return {"n": 0}
    idx = rng.randint(0, len(d), size=(N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return {"n": int(len(d)), "off": float(np.mean(a)), "on": float(np.mean(b)),
            "delta": float(d.mean()),
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}


def immobile_flags(st):
    """Per episode: 1 if some full window of IMMOBILE_WINDOW primitives moved < IMMOBILE_M."""
    roll = st.groupby("episode_id").sim_displacement_m.transform(
        lambda s: s.rolling(IMMOBILE_WINDOW, min_periods=IMMOBILE_WINDOW).sum())
    return (roll < IMMOBILE_M).groupby(st.episode_id).any().astype(int)


def _events(st):
    """Maximal collision runs: rows (episode_id, start, end) in step_index units."""
    c = st.engine_collision_flag.to_numpy() > 0
    eid = st.episode_id.to_numpy()
    new_ep = np.r_[True, eid[1:] != eid[:-1]]
    start = c & (new_ep | ~np.r_[False, c[:-1]])
    end = c & (np.r_[eid[1:] != eid[:-1], True] | ~np.r_[c[1:], False])
    si = st.step_index.to_numpy()
    return pd.DataFrame({"episode_id": eid[start], "start": si[start], "end": si[end]})


def fragmentation(st):
    ev = _events(st)
    owned = st.guard_action.isin(GUARD_OWNED).to_numpy() if "guard_action" in st else None
    frag = np.zeros(len(ev), dtype=bool)
    if owned is not None and len(ev):
        pos = {k: i for i, k in enumerate(zip(st.episode_id.to_numpy(), st.step_index.to_numpy()))}
        prev = ev.groupby("episode_id").end.shift()
        for i, (e, s, p) in enumerate(zip(ev.episode_id, ev.start, prev)):
            if pd.isna(p) or s - p - 1 > FRAGMENT_GAP:
                continue
            lo, hi = pos[(e, int(p))] + 1, pos[(e, int(s))]
            frag[i] = owned[lo:hi].any()
    length = (ev.end - ev.start + 1).to_numpy()
    return ev.assign(fragment=frag), length


def compare(off_dir, on_dir, agent):
    rng = np.random.RandomState(SEED)
    ep_off, st_off = _read(off_dir)
    ep_on, st_on = _read(on_dir)
    shared = ep_on.index.intersection(ep_off.index)
    ep_off, ep_on = ep_off.loc[shared], ep_on.loc[shared]
    st_off = st_off[st_off.episode_id.isin(shared)]
    st_on = st_on[st_on.episode_id.isin(shared)]
    s_off, s_on = _num(ep_off, "success").astype(int), _num(ep_on, "success").astype(int)
    out = {"off_dir": os.path.relpath(off_dir, REPO), "on_dir": os.path.relpath(on_dir, REPO),
           "episodes": int(len(shared))}

    # a. where the SR gain comes from
    steps_off = _num(ep_off, "step_count")
    cls = pd.Series("other", index=shared)
    cap = DECISION_CAP.get(agent)
    if cap is not None:
        dec = st_off.groupby("episode_id").is_decision.sum().reindex(shared, fill_value=0)
        cls[dec >= cap] = "budget"
    cls[steps_off >= MAX_STEPS] = "timeout"
    cls[_num(ep_off, "stuck_flag") > 0] = "stuck"
    net = int((s_on - s_off).sum())
    by = {}
    for name in ("stuck", "timeout", "budget", "other"):
        m = cls == name
        if not m.any():
            continue
        g, l = int(((s_off == 0) & (s_on == 1) & m).sum()), int(((s_off == 1) & (s_on == 0) & m).sum())
        by[name] = {"episodes": int(m.sum()), "sr_off": float(s_off[m].mean()),
                    "sr_on": float(s_on[m].mean()), "gained": g, "lost": l,
                    "share_of_net_gain": (g - l) / net if net else None}
    not_stuck = cls != "stuck"
    out["sr_gain_by_unguarded_state"] = {
        "classes": by, "net_gained_episodes": net,
        "sr_change_on_not_stuck_episodes": _paired(s_off[not_stuck], s_on[not_stuck], rng)}

    # b. residual collisions, fragmentation, contact per metre
    coll = st_on[st_on.engine_collision_flag > 0]
    if "guard_action" in coll:
        counts = coll.guard_action.value_counts()
        total = int(counts.sum())
        out["residual_collisions"] = {
            "total_steps": total,
            "by_source": {k: int(v) for k, v in counts.items()},
            "share_by_source": {k: float(v) / total for k, v in counts.items()} if total else {}}
    ev_on, len_on = fragmentation(st_on)
    ev_off, len_off = fragmentation(st_off.assign(guard_action="pass"))
    n = float(len(shared))
    out["fragmentation"] = {
        "gap_primitives": FRAGMENT_GAP,
        "events_per_episode_off": len(ev_off) / n,
        "events_per_episode_on": len(ev_on) / n,
        "fragment_share_on": float(ev_on.fragment.mean()) if len(ev_on) else None,
        "events_per_episode_on_merged": float((~ev_on.fragment).sum()) / n,
        "event_length_mean_off": float(len_off.mean()) if len(len_off) else None,
        "event_length_mean_on": float(len_on.mean()) if len(len_on) else None,
        "event_length_median_off": float(np.median(len_off)) if len(len_off) else None,
        "event_length_median_on": float(np.median(len_on)) if len(len_on) else None,
    }
    metres_off, metres_on = _num(ep_off, "path_length_m").sum(), _num(ep_on, "path_length_m").sum()
    out["contact_per_metre"] = {
        "off": float(_num(ep_off, "collision_count").sum() / metres_off) if metres_off else None,
        "on": float(_num(ep_on, "collision_count").sum() / metres_on) if metres_on else None}

    # c. length
    out["length"] = {
        "steps_per_episode": _paired(steps_off, _num(ep_on, "step_count"), rng),
        "timeout_share": _paired((steps_off >= MAX_STEPS).astype(int),
                                 (_num(ep_on, "step_count") >= MAX_STEPS).astype(int), rng)}

    # e. a stuck measure the stall veto does not satisfy by construction
    im_off = immobile_flags(st_off).reindex(shared, fill_value=0)
    im_on = immobile_flags(st_on).reindex(shared, fill_value=0)
    out["immobile"] = {"window_primitives": IMMOBILE_WINDOW, "max_metres": IMMOBILE_M,
                       "share": _paired(im_off, im_on, rng),
                       "proxy_stuck_share": _paired(_num(ep_off, "stuck_flag") > 0,
                                                    _num(ep_on, "stuck_flag") > 0, rng)}
    return out


def unguarded_immobile():
    out = {}
    for d in sorted(glob.glob(os.path.join(REPO, "results", "*", "*", "*"))):
        agent, bench, arm = d.split(os.sep)[-3:]
        if arm not in ("block", "slide") or not os.path.exists(os.path.join(d, "episodes.csv")):
            continue
        key = "/".join((agent, bench, arm))
        if not os.path.exists(os.path.join(d, "steps.csv.gz")):   # e.g. a migrated arm
            out[key] = {"immobile_share": "UNAVAILABLE"}
            continue
        ep, st = _read(d)
        im = immobile_flags(st).reindex(ep.index, fill_value=0)
        out[key] = {
            "immobile_share": float(im.mean()),
            "proxy_stuck_share": float((_num(ep, "stuck_flag") > 0).mean()),
            "timeout_share": float((_num(ep, "step_count") >= MAX_STEPS).mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results", "summary", "guard_mechanism.json"))
    args = ap.parse_args()
    arms = {}
    for on_dir in sorted(glob.glob(os.path.join(REPO, "results", "*", "*", "*_guard-*"))):
        agent, bench, arm = on_dir.split(os.sep)[-3:]
        regime = arm.split("_guard-")[0]
        if regime not in ("block", "slide") or not os.path.exists(os.path.join(on_dir, "episodes.csv")):
            continue   # the Tryout combo arms are compared in guard_effect.py
        print("  " + "/".join((agent, bench, arm)), flush=True)
        arms["/".join((agent, bench, arm))] = compare(
            os.path.join(REPO, "results", agent, bench, regime), on_dir, agent)
    out = {"guarded": arms, "unguarded": unguarded_immobile()}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote {} ({} guarded arms)".format(args.out, len(arms)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
