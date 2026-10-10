#!/usr/bin/env python3
"""P4 -- among decision steps whose primitive led to a collision, how much of the agent's
own score-margin budget eps would a re-ranking layer need to spend to reach a safe
alternative? Per (agent, benchmark), `block` regime.

Two tiers, two questions behind the same curve (paper section on tiers; tab_headroom's
$^{\\ast}$ footnote):

  T2 (a spatial candidate set, read straight off `min_eps_for_safe_alt` -- the smallest
  score-margin eps at which some non-chosen candidate is both within eps of the chosen
  one's score AND geometrically clear, computed live by the worker):
    availability(eps)         = P(margin <= eps | risky)              -- an alternative
                                 exists at all, safe or not
    headroom_unconditional(eps) = P(min_eps_for_safe_alt <= eps | risky)
    headroom_conditional(eps)   = headroom_unconditional / availability, i.e. P(safe |
                                 an alternative within eps exists) -- isolates the
                                 "safety" half of the question from the "availability" half

  T1 (a fixed 4-way {stop, forward, left, right} categorical, no candidate positions):
    turn and stop are collision-free by construction (they cannot produce an engine
    collision), so *any* non-chosen candidate at a risky (necessarily forward) decision is
    already safe -- there is no separate geometric check to run. headroom is then just
    availability: the margin between the chosen action and the best of the other three.
    headroom_unconditional == availability and headroom_conditional == 1.0 by
    construction; this is score headroom for a one-step veto, not re-ranking headroom.

Usage: analysis/headroom.py [--out results/summary/headroom.json]
Modern Python (3.9+, pandas, numpy); never imported by the harness.

Ported from the pre-release analyze_e5_eps_sweep.py (per-agent RUNS dict,
ETPNav only) generalised to walk every landed `block`-regime arm and to add the T1 branch
(the old script's `internal_*` columns were ETPNav-only).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "vlnsafe"))
sys.path.insert(0, REPO)
from vlnsafe import config as cfg  # noqa: E402

BENCHMARKS = ("r2r", "rxr")
REGIME = "block"
EPS_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

_NUMERIC_COLS = [
    "step_index", "is_decision", "engine_collision_flag",
    "prob_stop", "prob_forward", "prob_left", "prob_right",
    "top2_margin", "min_eps_for_safe_alt", "safe_alt_exists_eps10",
]


def _agent_ids():
    paths = glob.glob(os.path.join(REPO, "configs", "agents", "*.yaml"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


def _load_steps(agent, benchmark):
    path = os.path.join(REPO, "results", agent, benchmark, REGIME, "steps.csv.gz")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["episode_id", "step_index"]).reset_index(drop=True)


def add_margin_unified(df):
    """Same derivation as analysis/introspection.py::add_unified_signals's margin_unified
    (T2's own top2_margin, else the gap between a T1 categorical's top two probabilities);
    duplicated rather than imported so this module stays a single, independently readable
    file."""
    df = df.copy()
    probs = df[["prob_stop", "prob_forward", "prob_left", "prob_right"]]
    has_categorical = probs.notna().all(axis=1)
    cat_margin = pd.Series(np.nan, index=df.index)
    if has_categorical.any():
        p = np.clip(probs.loc[has_categorical].to_numpy(), 1e-12, 1.0)
        top2 = np.sort(p, axis=1)[:, -2:]
        cat_margin.loc[has_categorical] = top2[:, 1] - top2[:, 0]
    df["margin_unified"] = df["top2_margin"].combine_first(cat_margin)
    return df


def add_label_k1(df):
    """label_k1[i] = 1 iff engine_collision_flag==1 at step i itself -- the decision at
    step i governs the primitive about to execute at step i."""
    df = df.copy()
    df["label_k1"] = (df["engine_collision_flag"].fillna(0) > 0).astype(int)
    return df


def sweep_t2(risky):
    curve = {}
    n_risky = len(risky)
    for eps in EPS_GRID:
        avail = risky["margin_unified"] <= eps
        n_avail = int(avail.sum())
        headroom_unconditional = float((risky["min_eps_for_safe_alt"] <= eps).mean()) if n_risky else None
        headroom_conditional = (
            float((risky.loc[avail, "min_eps_for_safe_alt"] <= eps).mean()) if n_avail else None
        )
        curve[eps] = {
            "availability_rate_among_risky": float(avail.mean()) if n_risky else None,
            "n_with_availability": n_avail,
            "headroom_rate_unconditional": headroom_unconditional,
            "headroom_rate_conditional_on_availability": headroom_conditional,
        }
    return curve


def sweep_t1(risky):
    """Safe by construction (module docstring): headroom == availability, and the
    conditional rate is 1.0 wherever an alternative is available at all."""
    curve = {}
    n_risky = len(risky)
    for eps in EPS_GRID:
        avail = risky["margin_unified"] <= eps
        n_avail = int(avail.sum())
        rate = float(avail.mean()) if n_risky else None
        curve[eps] = {
            "availability_rate_among_risky": rate,
            "n_with_availability": n_avail,
            "headroom_rate_unconditional": rate,
            "headroom_rate_conditional_on_availability": 1.0 if n_avail else None,
        }
    return curve


def analyze_arm(agent, benchmark, tier):
    df = _load_steps(agent, benchmark)
    if df is None:
        return None
    df = add_margin_unified(df)
    df = add_label_k1(df)
    dec = df[df["is_decision"] == 1]
    risky = dec[dec["label_k1"] == 1]
    n_risky = len(risky)

    curve = sweep_t2(risky) if tier == "T2" else sweep_t1(risky)
    return {
        "regime": REGIME,
        "tier": tier,
        "n_decision_steps": int(len(dec)),
        "n_risky_decisions": int(n_risky),
        "availability_rate_all_decisions_eps10pct": float((dec["margin_unified"] <= 0.10).mean()) if len(dec) else None,
        "eps_sweep": {str(eps): v for eps, v in curve.items()},
    }


def collect():
    out = {}
    for agent in _agent_ids():
        tier = cfg.load_agent(agent)["tier"]
        agent_block = {}
        for benchmark in BENCHMARKS:
            arm = analyze_arm(agent, benchmark, tier)
            if arm is not None:
                agent_block[benchmark] = arm
        if agent_block:
            out[agent] = agent_block
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "summary", "headroom.json"))
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
