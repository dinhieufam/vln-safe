#!/usr/bin/env python3
"""P1 (safety profile) and P2 (regime + predicate validity) for one arm, from its
episodes.csv.

P1 — outcome x contact: the joint success/collision table, dirty-success rate with a
Wilson interval, collisions per dirty success, the share of all collision events that
land inside successful episodes, min-clearance by outcome with a Mann-Whitney p, time in
risk, nc-rate, stuck rate.

P2 — regime + predicate: the proxy-vs-engine disagreement on the same episodes and
kappa-hat-miss (share of engine-contact episodes the displacement proxy reports clean).
The regime delta itself (block vs slide) is computed by analysis/collect.py, which calls
this for both arms.

Usage: analysis/profile.py --agent etpnav --benchmark r2r --regime block
       analysis/profile.py results/etpnav/r2r/block/episodes.csv
Modern Python (3.9+, pandas, scipy); never imported by the harness.

Ported from vln_research/pipeline/analysis/compute_paper_stats.py (arm_stats +
scene_breakdown), with the four hard-coded arm paths replaced by --agent/--benchmark/
--regime and the column names updated to docs/schema.md (engine = collision_count,
proxy = proxy_collision_count).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _num(df, col, mask):
    v = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna()
    return {"mean": float(v.mean()), "median": float(v.median()), "n": int(len(v))}


def arm_profile(df):
    succ = df["success"].astype(float).astype(bool)
    engine = pd.to_numeric(df["collision_count"], errors="coerce").fillna(0)
    coll = engine > 0
    n, ns = len(df), int(succ.sum())

    joint = {
        "clean_success": int((succ & ~coll).sum()),
        "dirty_success": int((succ & coll).sum()),
        "clean_failure": int((~succ & ~coll).sum()),
        "dirty_failure": int((~succ & coll).sum()),
    }
    lo, hi = wilson(joint["dirty_success"], ns)
    out = {
        "n_episodes": n, "n_success": ns,
        "joint_counts": joint,
        "joint_rates": {k: v / n for k, v in joint.items()},
        "p1": {
            "dirty_success_rate": joint["dirty_success"] / ns if ns else float("nan"),
            "dirty_success_ci95": [lo, hi],
            "collisions_per_success": float(engine[succ].mean()) if ns else float("nan"),
            "collisions_per_dirty_success": float(engine[succ & coll].mean()) if joint["dirty_success"] else float("nan"),
            "collisions_per_dirty_success_median": float(engine[succ & coll].median()) if joint["dirty_success"] else float("nan"),
            "share_of_collisions_in_successes": float(engine[succ].sum() / engine.sum()) if engine.sum() else float("nan"),
            "min_clearance_success": _num(df, "min_clearance_m", succ),
            "min_clearance_failure": _num(df, "min_clearance_m", ~succ),
            "time_in_risk_success": _num(df, "time_in_risk_rate", succ),
            "time_in_risk_failure": _num(df, "time_in_risk_rate", ~succ),
            "nc_rate_success": _num(df, "nc_rate", succ),
            "nc_rate_failure": _num(df, "nc_rate", ~succ),
            "nc_rate_mean": float(pd.to_numeric(df["nc_rate"], errors="coerce").mean()),
            "time_in_risk_rate_mean": float(pd.to_numeric(df["time_in_risk_rate"], errors="coerce").mean()),
            "stuck_rate": float(pd.to_numeric(df["stuck_flag"], errors="coerce").fillna(0).mean()),
        },
    }

    a = pd.to_numeric(df.loc[succ, "min_clearance_m"], errors="coerce").dropna()
    b = pd.to_numeric(df.loc[~succ, "min_clearance_m"], errors="coerce").dropna()
    if len(a) > 10 and len(b) > 10:
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        out["p1"]["min_clearance_success_vs_failure"] = {
            "u": float(u.statistic), "p": float(u.pvalue),
            "delta_mean_m": float(a.mean() - b.mean()),
        }

    proxy = pd.to_numeric(df["proxy_collision_count"], errors="coerce").fillna(0) > 0
    out["p2"] = {
        "dirty_success_rate_proxy": float((succ & proxy).sum() / ns) if ns else float("nan"),
        "proxy_engine_disagreement_rate": float((proxy != coll).mean()),
        "kappa_miss": float((~proxy & coll).sum() / coll.sum()) if coll.sum() else float("nan"),
        "n_engine_contact_episodes": int(coll.sum()),
        "n_missed_by_proxy": int((~proxy & coll).sum()),
    }
    return out


def scene_breakdown(df):
    succ = df["success"].astype(float).astype(bool)
    coll = pd.to_numeric(df["collision_count"], errors="coerce").fillna(0) > 0
    rows = []
    for scene, sub in df.assign(_s=succ, _c=coll).groupby("scene_id"):
        ns = int(sub["_s"].sum())
        if ns < 10:
            continue
        rows.append({"scene": str(scene), "n_success": ns,
                     "dirty_success_rate": float((sub["_s"] & sub["_c"]).sum() / ns)})
    rows.sort(key=lambda r: r["dirty_success_rate"])
    vals = [r["dirty_success_rate"] for r in rows]
    return {"per_scene": rows, "n_scenes_reported": len(rows),
            "min": min(vals) if vals else None, "max": max(vals) if vals else None,
            "median": float(np.median(vals)) if vals else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episodes_csv", nargs="?")
    ap.add_argument("--agent")
    ap.add_argument("--benchmark")
    ap.add_argument("--regime")
    args = ap.parse_args()

    if args.episodes_csv:
        path = args.episodes_csv
    elif args.agent and args.benchmark and args.regime:
        path = os.path.join(REPO, "results", args.agent, args.benchmark, args.regime, "episodes.csv")
    else:
        ap.error("give an episodes.csv path or --agent --benchmark --regime")

    df = pd.read_csv(path).drop_duplicates(subset=["episode_id"], keep="first")
    out = arm_profile(df)
    out["scenes"] = scene_breakdown(df)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
