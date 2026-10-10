#!/usr/bin/env python3
"""Precision and recall of the guard's depth veto per memory length, from
scripts/replay_guard_memory.py sidecars.

Population: executed forward steps of an unguarded arm that do NOT directly follow a stalled
forward (displacement < 1 cm) -- those are the stall veto's job (guard `stall`), and they
collide ~100 % of the time whatever the depth says. What is left is the depth veto's domain:
first contacts, and every clean forward step it might wrongly veto.

For each memory column (length M, optionally range-gated to R m: M<M>_r<R>) and threshold t:
veto = clearance < t.
  precision       P(collided | veto)
  recall          P(veto | collided)
  false vetoes/ep vetoes on steps that did not collide, per episode

Usage: analysis/guard_memory.py [--root data/scratch/memdiag] [--out results/summary/guard_memory.json]
"""
import argparse
import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
THRESHOLDS = (0.0, 0.05)
STALL_M = 0.01


def summarize(df):
    df = df.sort_values(["episode_id", "step_index"])
    prev = df.groupby("episode_id").shift(1)
    after_stall = (prev.step_index == df.step_index - 1) & (prev.displacement_m < STALL_M)
    dom = df[~after_stall]
    n_ep = df.episode_id.nunique()
    out = {"episodes": int(n_ep), "forward_steps": int(len(df)),
           "depth_domain_steps": int(len(dom)), "depth_domain_collisions": int(dom.collided.sum()),
           "memory": {}}
    for col in (c for c in df.columns if c.startswith("clear_m")):
        label = "M" + col[len("clear_m"):]     # M8, or M8_r1.5 for a range-gated memory
        c = pd.to_numeric(dom[col], errors="coerce")
        for t in THRESHOLDS:
            veto = c < t
            hit = dom.collided == 1
            out["memory"]["{}_t{}".format(label, t)] = {
                "vetoes_per_episode": float(veto.sum() / n_ep),
                "precision": float(hit[veto].mean()) if veto.any() else None,
                "recall": float(veto[hit].mean()) if hit.any() else None,
                "false_vetoes_per_episode": float((veto & ~hit).sum() / n_ep),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "data", "scratch", "memdiag"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "summary", "guard_memory.json"))
    a = ap.parse_args()
    res = {}
    for path in sorted(glob.glob(os.path.join(a.root, "*", "guard_memory.csv.gz"))):
        agent = os.path.basename(os.path.dirname(path))
        if agent.startswith("test"):
            continue
        res[agent] = summarize(pd.read_csv(path, dtype={"episode_id": str}))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    for agent, r in res.items():
        print("{}: {} episodes, {} depth-domain forward steps, {} collide".format(
            agent, r["episodes"], r["depth_domain_steps"], r["depth_domain_collisions"]))
        for k, v in r["memory"].items():
            print("  {:10s} vetoes/ep {:5.2f}  precision {}  recall {}  false vetoes/ep {:5.2f}".format(
                k, v["vetoes_per_episode"],
                "  -  " if v["precision"] is None else "{:.3f}".format(v["precision"]),
                "  -  " if v["recall"] is None else "{:.3f}".format(v["recall"]),
                v["false_vetoes_per_episode"]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
