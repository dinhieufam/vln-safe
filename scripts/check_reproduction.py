#!/usr/bin/env python3
"""Reproduction gate: an arm's SR / SPL vs the agent's published number, with the
per-family tolerance from its config (the reproduction gate). Pure — reads episodes.csv, no habitat.

    python scripts/check_reproduction.py results/etpr1/r2r/slide

Exit 0 when SR is within `reproduction_tolerance_sr` of `published.<benchmark>.sr` (or
when there is no public number to gate against — `sr: null`); exit 1 on a miss.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vlnsafe import config as cfg  # noqa: E402


def _mean(rows, col):
    vals = [float(r[col]) for r in rows if r.get(col) not in ("", "UNAVAILABLE", None)]
    return (sum(vals) / len(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir", help="results/<agent>/<benchmark>/<regime>")
    ap.add_argument("--episodes-csv", default=None)
    args = ap.parse_args()

    parts = os.path.normpath(args.arm_dir).split(os.sep)
    agent_id, benchmark, regime = parts[-3], parts[-2], parts[-1]
    ep_csv = args.episodes_csv or os.path.join(args.arm_dir, "episodes.csv")
    if not os.path.exists(ep_csv):
        raise SystemExit("no episodes.csv at {}".format(ep_csv))

    with open(ep_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    sr = _mean(rows, "success")
    spl = _mean(rows, "spl")
    sr_pct = None if sr is None else 100.0 * sr
    spl_pct = None if spl is None else 100.0 * spl

    agent = cfg.load_agent(agent_id)
    published = (agent.get("published") or {}).get(benchmark) or {}
    tol = float(agent.get("reproduction_tolerance_sr", 1.0))
    target = published.get("sr")

    print("arm         : {} / {} / {}   (n={})".format(agent_id, benchmark, regime, len(rows)))
    print("reproduced  : SR {:.2f}   SPL {}".format(
        sr_pct or 0.0, "n/a" if spl_pct is None else "{:.2f}".format(spl_pct)))
    if target is None:
        print("published   : none for this benchmark -> no reproduction gate, skipped")
        return 0
    delta = abs(sr_pct - target)
    ok = delta <= tol
    print("published   : SR {:.2f}  ({})".format(target, published.get("source", "")))
    print("delta SR    : {:.2f}   tolerance +/-{:.1f}   -> {}".format(
        delta, tol, "PASS" if ok else "FAIL"))
    if regime != published.get("regime"):
        print("note        : arm regime {!r} != published regime {!r}".format(
            regime, published.get("regime")))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
