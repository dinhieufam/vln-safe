#!/usr/bin/env python3
"""P2 regime effect with paired statistics. For every (agent,
benchmark) with both `block` and `slide` arms, paired on episode_id:

  - `success`: McNemar's exact test on the per-episode success pairs, and the paired SR
    change slide - block with a 95% bootstrap interval.
  - `dirty_success`: DSR = P(contact | success) per regime and the change slide - block
    with a 95% bootstrap interval over episodes (each resample recomputes both
    conditional rates, since the success sets differ between regimes).
  - `length_controlled`: does `block` lower DSR only because its successes are shorter?
    Successful episodes of both regimes are binned into quartiles of step_count (pooled
    edges); DSR per bin and regime, and the bin-weighted difference (weights = pooled
    successes per bin), with a bootstrap interval. Also the mean length of a success and
    collision steps per 100 primitives within successes.
  - `survivorship`: of the episodes that succeed under `slide`, the share that fail under
    `block`, split by whether the `slide` success was dirty or clean. If contact under
    `block` turns would-be successes into failures, dirty ones fail far more often.

Usage: analysis/regime_stats.py [--out results/summary/regime_stats.json]
Modern Python (pandas, numpy, scipy); never imported by the harness.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
N_BOOT = 2000
SEED = 20260927
N_BINS = 4


def _read(d):
    ep = pd.read_csv(os.path.join(d, "episodes.csv"), dtype={"episode_id": str})
    ep = ep.drop_duplicates("episode_id", keep="last").set_index("episode_id")
    return pd.DataFrame({
        "success": pd.to_numeric(ep.success, errors="coerce").fillna(0).astype(int),
        "dirty": (pd.to_numeric(ep.collision_count, errors="coerce").fillna(0) > 0).astype(int),
        "steps": pd.to_numeric(ep.step_count, errors="coerce"),
        "coll": pd.to_numeric(ep.collision_count, errors="coerce").fillna(0)})


def _dsr(df):
    s = df[df.success == 1]
    return float(s.dirty.mean()) if len(s) else np.nan


def _stratified(b, s, edges):
    """Bin-weighted DSR difference slide - block, weights = pooled successes per bin."""
    diff, weight = 0.0, 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sb = b[(b.success == 1) & (b.steps > lo) & (b.steps <= hi)]
        ss = s[(s.success == 1) & (s.steps > lo) & (s.steps <= hi)]
        if len(sb) and len(ss):
            w = len(sb) + len(ss)
            diff += w * (ss.dirty.mean() - sb.dirty.mean())
            weight += w
    return diff / weight if weight else np.nan


def _ci(vals):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def compare(block_dir, slide_dir):
    rng = np.random.RandomState(SEED)
    b, s = _read(block_dir), _read(slide_dir)
    shared = b.index.intersection(s.index)
    b, s = b.loc[shared], s.loc[shared]
    n = len(shared)

    only_slide = int(((s.success == 1) & (b.success == 0)).sum())
    only_block = int(((b.success == 1) & (s.success == 0)).sum())
    k = only_slide + only_block
    p = binomtest(min(only_slide, only_block), k, 0.5).pvalue if k else 1.0
    d = (s.success - b.success).to_numpy(float)

    succ = pd.concat([b[b.success == 1].steps, s[s.success == 1].steps])
    edges = np.unique(np.quantile(succ, np.linspace(0, 1, N_BINS + 1)))
    edges[0] = -np.inf

    idx = rng.randint(0, n, size=(N_BOOT, n))
    sr_boot, dsr_boot, strat_boot = [], [], []
    for row in idx:
        bb, ss = b.iloc[row], s.iloc[row]
        sr_boot.append(d[row].mean())
        dsr_boot.append(_dsr(ss) - _dsr(bb))
        strat_boot.append(_stratified(bb, ss, edges))

    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sb = b[(b.success == 1) & (b.steps > lo) & (b.steps <= hi)]
        ss = s[(s.success == 1) & (s.steps > lo) & (s.steps <= hi)]
        bins.append({"steps_upper": float(hi), "n_block": int(len(sb)), "n_slide": int(len(ss)),
                     "dsr_block": float(sb.dirty.mean()) if len(sb) else None,
                     "dsr_slide": float(ss.dirty.mean()) if len(ss) else None})

    def within_success(df):
        x = df[df.success == 1]
        return {"mean_steps": float(x.steps.mean()),
                "collisions_per_100_primitives": float(100.0 * x.coll.sum() / x.steps.sum())}

    ss_ok = s.success == 1
    fail_dirty = b.success[ss_ok & (s.dirty == 1)] == 0
    fail_clean = b.success[ss_ok & (s.dirty == 0)] == 0
    return {
        "survivorship": {
            "slide_dirty_successes": int(len(fail_dirty)),
            "slide_clean_successes": int(len(fail_clean)),
            "block_fail_share_if_slide_dirty": float(fail_dirty.mean()) if len(fail_dirty) else None,
            "block_fail_share_if_slide_clean": float(fail_clean.mean()) if len(fail_clean) else None},
        "episodes": int(n),
        "success": {"block": float(b.success.mean()), "slide": float(s.success.mean()),
                    "delta": float(d.mean()), "ci95": _ci(sr_boot),
                    "only_slide": only_slide, "only_block": only_block, "mcnemar_exact_p": float(p)},
        "dirty_success": {"block": _dsr(b), "slide": _dsr(s), "delta": _dsr(s) - _dsr(b),
                          "ci95": _ci(dsr_boot)},
        "length_controlled": {"bins": bins, "stratified_delta": _stratified(b, s, edges),
                              "ci95": _ci(strat_boot),
                              "successes_block": within_success(b),
                              "successes_slide": within_success(s)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results", "summary", "regime_stats.json"))
    args = ap.parse_args()
    out = {}
    for block_dir in sorted(glob.glob(os.path.join(REPO, "results", "*", "*", "block"))):
        slide_dir = os.path.join(os.path.dirname(block_dir), "slide")
        if not all(os.path.exists(os.path.join(x, "episodes.csv")) for x in (block_dir, slide_dir)):
            continue
        agent, bench = block_dir.split(os.sep)[-3:-1]
        out["/".join((agent, bench))] = compare(block_dir, slide_dir)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote {} ({} agent/benchmark pairs)".format(args.out, len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
