#!/usr/bin/env python3
"""Guard runtime overhead: wall clock per executed step, guard off vs on.

Reads the timing.csv sidecars that `vlnsafe run --timing` writes (one row per step: policy,
logged clearance predictor, guard, simulator, whole step, in seconds) for two diagnostic arms
on the same episode slice, e.g. ETP-R1 `block` on the first 50 R2R val-unseen episodes:

    analysis/runtime.py <off_dir> <on_dir> [--out results/summary/runtime.json]

Reports mean / median / p95 ms per step for every component, the guard's share of the
guarded step, and mean seconds per episode. The first step of a shard is dropped: its
policy time includes the worker's model load. The two arms' trajectories differ once the
guard intervenes, so per-step components, not episode totals, carry the comparison.

Wall clock on a shared machine: report it as measured, with the GPU and load it ran under.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

_COLS = ["policy_s", "predictor_s", "guard_s", "sim_s", "step_s"]


def _stats(t):
    out = {}
    for c in _COLS:
        ms = t[c].to_numpy() * 1000.0
        out[c.replace("_s", "_ms")] = {
            "mean": float(ms.mean()), "median": float(np.median(ms)),
            "p95": float(np.percentile(ms, 95))}
    return out


def _arm(path):
    t = pd.read_csv(os.path.join(path, "timing.csv"))
    t = t.iloc[1:]   # first step: worker model load inside the first policy call
    per_ep = t.groupby("episode_id")["step_s"].sum()
    with open(os.path.join(path, "manifest.json")) as f:
        manifest = json.load(f)
    return t, {
        "dir": path, "episodes": int(t["episode_id"].nunique()), "steps": int(len(t)),
        "gpu": manifest.get("gpu"), "guard": (manifest.get("guard") or {}).get("name"),
        "per_step": _stats(t),
        "episode_s_mean": float(per_ep.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("off_dir")
    ap.add_argument("on_dir")
    ap.add_argument("--out", default="results/summary/runtime.json")
    args = ap.parse_args()
    _, off = _arm(args.off_dir)
    t_on, on = _arm(args.on_dir)
    guard_ms = on["per_step"]["guard_ms"]["mean"]
    step_ms = on["per_step"]["step_ms"]["mean"]
    vetoed = t_on["guard_action"].isin(["abort", "stall", "stop"])
    summary = {
        "off": off, "on": on,
        "guard_ms_per_step": guard_ms,
        "guard_share_of_step": guard_ms / step_ms,
        "guard_ms_on_veto_steps": float(t_on.loc[vetoed, "guard_s"].mean() * 1000.0),
        "guard_ms_on_pass_steps": float(t_on.loc[~vetoed, "guard_s"].mean() * 1000.0),
        "loadavg_at_write": open("/proc/loadavg").read().split()[:3],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("guard {:.2f} ms/step ({:.1%} of a guarded step, {:.1f} ms); veto steps {:.2f} ms"
          .format(guard_ms, summary["guard_share_of_step"], step_ms,
                  summary["guard_ms_on_veto_steps"]))
    print("step ms off/on: {:.1f} / {:.1f}; policy {:.1f} / {:.1f}; sim {:.1f} / {:.1f}".format(
        off["per_step"]["step_ms"]["mean"], step_ms,
        off["per_step"]["policy_ms"]["mean"], on["per_step"]["policy_ms"]["mean"],
        off["per_step"]["sim_ms"]["mean"], on["per_step"]["sim_ms"]["mean"]))


if __name__ == "__main__":
    main()
