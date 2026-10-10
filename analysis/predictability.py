#!/usr/bin/env python3
"""P5 -- Phase 2a: the AUROC table in analysis/introspection.py
says collision is predictable from the external depth-projected clearance
(`predicted_clearance_m`, `geometry_inv` in that module's SIGNALS -- internal scores stay
near chance on every agent). This module asks the three questions AUROC cannot answer,
because a veto layer needs an operating point, not a threshold-free summary:

  1. Lead time -- at a fixed intervention budget, how many primitive steps before the
     first contact of a collision event does the geometry signal first cross into
     "flagged", persisting to contact? (This is the warning a veto layer would actually
     get, not just "is the signal informative on average".)
  2. Operating points -- P(collision | flagged) and P(flagged | collision) at fixed flag
     rates among decision steps (1%, 5%, 10%) -- the flag rate *is* the intervention
     budget a guard layer spends, so this is the number that sets its threshold
, not AUROC.
  3. Collision clustering -- what fraction of engine-collision steps belong to a streak
     already in progress, using rollout.py's own streak counter
     (`collision_event_count` vs `collision_count`, rollout.py:104-119) rather than
     recomputing it. A layer that prevents only the first contact of a streak is worth
     far more than the raw per-step collision rate suggests -- this is the reading
     behind CMA's 87.9% per-step collision rate.

Signal scored: `geometry_inv` only (score = -predicted_clearance_m, same sign convention
as analysis/introspection.py::SIGNALS). Internal signals are out of scope here -- section
2 of the plan already establishes they are near chance on every agent; a veto layer built
on this analysis uses the external predictor.

Usage: analysis/predictability.py [--out results/summary/predictability.json]
Modern Python (3.9+, pandas, numpy); never imported by the harness.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

BENCHMARKS = ("r2r", "rxr")
REGIME = "block"           # collisions are physical only in this regime
HORIZONS = (1, 3, 5)       # same primitive-step windows as introspection.py
FLAG_RATES = (0.01, 0.05, 0.10)

_NUMERIC_COLS = [
    "step_index", "is_decision", "engine_collision_flag", "predicted_clearance_m",
]
_EPISODE_NUMERIC_COLS = ["collision_count", "collision_event_count"]


def _agent_ids():
    paths = glob.glob(os.path.join(REPO, "configs", "agents", "*.yaml"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


def attach_replay_geometry(df, sidecar_paths):
    """Same join as analysis/introspection.py::attach_replay_geometry (T1's
    `predicted_clearance_m` is recovered from scripts/replay_geometry.py sidecars, never
    from a re-run); duplicated rather than imported so this module stays a single,
    independently readable file, per the other analysis/*.py scripts' own convention."""
    if not sidecar_paths or df["predicted_clearance_m"].notna().any():
        return df, 0
    parts = [pd.read_csv(p, dtype={"episode_id": str}) for p in sidecar_paths]
    geo = pd.concat(parts, ignore_index=True)
    geo["predicted_clearance_m"] = pd.to_numeric(geo["predicted_clearance_m"], errors="coerce")
    geo = geo.drop_duplicates(["episode_id", "step_index"])
    geo = geo.assign(_eid=geo["episode_id"].astype(str)).drop(columns=["episode_id"])
    out = df.assign(_eid=df["episode_id"].astype(str)).drop(columns=["predicted_clearance_m"])
    out = out.merge(geo, on=["_eid", "step_index"], how="left").drop(columns=["_eid"])
    return out, int(out["predicted_clearance_m"].notna().sum())


def _replay_sidecars(agent, benchmark):
    root = os.path.join(REPO, "data", "scratch", agent, benchmark, REGIME, "geometry_replay")
    return sorted(glob.glob(os.path.join(root, "shard*", "geometry.csv.gz")))


def _load_steps(agent, benchmark):
    path = os.path.join(REPO, "results", agent, benchmark, REGIME, "steps.csv.gz")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df, _ = attach_replay_geometry(df, _replay_sidecars(agent, benchmark))
    return df.sort_values(["episode_id", "step_index"]).reset_index(drop=True)


def _load_episodes(agent, benchmark):
    path = os.path.join(REPO, "results", agent, benchmark, REGIME, "episodes.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in _EPISODE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_horizon_labels(df, horizons=HORIZONS):
    """Same definition as analysis/introspection.py::add_horizon_labels: label_k[i] = 1
    if engine_collision_flag==1 at any primitive step in [i, i+k-1] of the same episode.
    Duplicated for the same reason as attach_replay_geometry above."""
    df = df.copy()
    coll = df["engine_collision_flag"].fillna(0)
    for k in horizons:
        label = coll.groupby(df["episode_id"]).transform(
            lambda s: s[::-1].rolling(k, min_periods=1).max()[::-1]
        )
        df["label_k{}".format(k)] = label.astype(int)
    return df


def add_geometry_score(df):
    """geometry_inv's scored value: -predicted_clearance_m, so higher = riskier, same
    sign convention as introspection.py::SIGNALS["geometry_inv"]."""
    df = df.copy()
    df["geometry_score"] = -1.0 * df["predicted_clearance_m"]
    return df


def collision_events(df):
    """One row per engine-collision streak, using rollout.py's own onset rule
    (rollout.py:104-108: a streak starts whenever engine_collision_flag rises from 0 to 1
    within an episode) -- not recomputed differently here, so this always agrees with
    episodes.csv's `collision_event_count`."""
    d = df[["episode_id", "step_index", "engine_collision_flag"]].copy()
    d["engine_collision_flag"] = d["engine_collision_flag"].fillna(0).astype(int)
    d = d.sort_values(["episode_id", "step_index"]).reset_index(drop=True)
    same_ep_prev = d["episode_id"].eq(d["episode_id"].shift(1))
    prev_flag = d["engine_collision_flag"].shift(1).fillna(0).astype(int)
    is_onset = (d["engine_collision_flag"] == 1) & ~(same_ep_prev & (prev_flag == 1))
    d["event_id"] = is_onset.cumsum()
    hits = d[d["engine_collision_flag"] == 1]
    events = (
        hits.groupby(["episode_id", "event_id"])
        .agg(onset_step_index=("step_index", "min"), length_steps=("step_index", "size"))
        .reset_index()
    )
    return events[["episode_id", "onset_step_index", "length_steps"]]


def _distribution_stats(arr):
    if arr.size == 0:
        return None
    hist = {str(k): int((arr == k).sum()) for k in range(5)}
    hist["5+"] = int((arr >= 5).sum())
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "histogram_primitive_steps": hist,
    }


def _lead_time_for_flag(dec, flagged, events):
    """For each collision event, walk backward from the last decision row at or before
    its onset while `flagged` stays true; lead time is how many primitive steps earlier
    than contact that persistent-flagged run began. An event whose nearest preceding
    decision row is not flagged is a miss at this operating point (excluded from the
    distribution, counted in detection_rate)."""
    flagged_dec = pd.DataFrame({
        "episode_id": dec["episode_id"].to_numpy(),
        "step_index": dec["step_index"].to_numpy(),
        "flagged": np.asarray(flagged, dtype=bool),
    }).sort_values(["episode_id", "step_index"])
    groups = {
        eid: (g["step_index"].to_numpy(), g["flagged"].to_numpy())
        for eid, g in flagged_dec.groupby("episode_id", sort=False)
    }
    lead_times = []
    for row in events.itertuples(index=False):
        arrs = groups.get(row.episode_id)
        if arrs is None:
            continue
        steps, flags = arrs
        idx = int(np.searchsorted(steps, row.onset_step_index, side="right")) - 1
        if idx < 0 or not flags[idx]:
            continue
        i = idx
        while i - 1 >= 0 and flags[i - 1]:
            i -= 1
        lead_times.append(int(row.onset_step_index - steps[i]))
    arr = np.array(lead_times, dtype=float)
    n_events = int(len(events))
    return {
        "n_events": n_events,
        "n_detected": int(arr.size),
        "detection_rate": float(arr.size / n_events) if n_events else None,
        "distribution_primitive_steps": _distribution_stats(arr),
    }


def analyze_operating_points(df, dec, events, flag_rates=FLAG_RATES, horizons=HORIZONS):
    s = dec["geometry_score"].dropna()
    n_dec = len(dec)
    out = {}
    for f in flag_rates:
        key = str(f)
        if len(s) == 0:
            out[key] = None
            continue
        thr_score = float(s.quantile(1.0 - f))
        flagged = (dec["geometry_score"] >= thr_score).fillna(False)
        n_flagged = int(flagged.sum())
        precision_by_horizon = {}
        for k in horizons:
            y = dec["label_k{}".format(k)]
            n_pos = int(y.sum())
            precision_by_horizon[k] = {
                "precision": float(y[flagged].mean()) if n_flagged else None,
                "recall": float(y[flagged].sum() / n_pos) if n_pos else None,
                "base_rate": float(y.mean()) if len(y) else None,
            }
        out[key] = {
            "threshold_geometry_score": thr_score,
            "threshold_clearance_m": -thr_score,
            "n_flagged": n_flagged,
            "actual_flag_rate": float(n_flagged / n_dec) if n_dec else None,
            "precision_by_horizon": precision_by_horizon,
            "lead_time": _lead_time_for_flag(dec, flagged, events),
        }
    return out


def streak_summary(episodes):
    """Read straight from rollout.py's own counters (episodes.csv columns
    `collision_count`, `collision_event_count`), never recomputed from steps, so this
    cannot disagree with what the harness itself logged."""
    total_steps = float(episodes["collision_count"].sum())
    total_events = float(episodes["collision_event_count"].sum())
    n_with_collision = int((episodes["collision_count"] > 0).sum())
    if total_steps == 0:
        return {
            "n_episodes": int(len(episodes)),
            "n_episodes_with_collision": 0,
            "total_collision_steps": 0,
            "total_collision_events": 0,
            "streak_continuation_fraction": None,
            "mean_event_length_steps": None,
        }
    return {
        "n_episodes": int(len(episodes)),
        "n_episodes_with_collision": n_with_collision,
        "total_collision_steps": int(total_steps),
        "total_collision_events": int(total_events),
        "streak_continuation_fraction": float((total_steps - total_events) / total_steps),
        "mean_event_length_steps": float(total_steps / total_events) if total_events else None,
    }


def analyze_arm(agent, benchmark):
    df = _load_steps(agent, benchmark)
    episodes = _load_episodes(agent, benchmark)
    if df is None or episodes is None:
        return None
    df = add_horizon_labels(df)
    df = add_geometry_score(df)
    dec = df[df["is_decision"] == 1].reset_index(drop=True)
    events = collision_events(df)
    geometry_available = bool(dec["geometry_score"].notna().any())
    return {
        "regime": REGIME,
        "n_rows": int(len(df)),
        "n_decision_steps": int(len(dec)),
        "geometry_available": geometry_available,
        "operating_points": (
            analyze_operating_points(df, dec, events) if geometry_available else None
        ),
        "collision_streaks": streak_summary(episodes),
    }


def collect():
    out = {}
    for agent in _agent_ids():
        agent_block = {}
        for benchmark in BENCHMARKS:
            arm = analyze_arm(agent, benchmark)
            if arm is not None:
                agent_block[benchmark] = arm
        if agent_block:
            out[agent] = agent_block
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "summary", "predictability.json"))
    args = ap.parse_args()

    out = collect()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print("wrote {} ({} agents)".format(args.out, len(out)))
    for agent in sorted(out):
        print("  {}: {}".format(agent, ", ".join(sorted(out[agent]))))


if __name__ == "__main__":
    main()
