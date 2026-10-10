#!/usr/bin/env python3
"""Proxy-vs-engine collision disagreement, at episode granularity.

What: for episodes where the physics engine (habitat's contact test) recorded
at least one collision, what share does the displacement-based proxy report
as clean? This is kappa-hat-miss, Table 6's validity check on the proxy the
harness falls back to for workers that cannot see engine contacts.

Who reads it: analysis/make_tables.py (Table 6, kappa-hat-miss column) reads
the value this script writes into results/summary/paper_numbers.json.

Input: a step log (results/<agent>/<benchmark>/<regime>/steps.csv[.gz]) with
one row per executed step and boolean engine/proxy collision columns.
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


def open_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    return opener(path, "rt", newline="")


def kappa_miss(steps_path, episode_col, engine_col, proxy_col):
    engine_hits = defaultdict(int)
    proxy_hits = defaultdict(int)
    with open_csv(steps_path) as f:
        for row in csv.DictReader(f):
            eid = row[episode_col]
            engine_hits[eid] += int(float(row[engine_col] or 0))
            proxy_hits[eid] += int(float(row[proxy_col] or 0))
    engine_contact = [e for e in engine_hits if engine_hits[e] > 0]
    missed = [e for e in engine_contact if proxy_hits[e] == 0]
    n = len(engine_contact)
    return {
        "n_engine_contact_episodes": n,
        "n_missed_by_proxy": len(missed),
        "kappa_miss": len(missed) / n if n else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("steps_csv", type=Path)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--episode-col", default="episode_id")
    ap.add_argument("--engine-col", default="engine_collision_flag")
    ap.add_argument("--proxy-col", default="proxy_collision_flag")
    ap.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "summary" / "paper_numbers.json",
    )
    args = ap.parse_args()

    result = kappa_miss(args.steps_csv, args.episode_col, args.engine_col, args.proxy_col)

    numbers = json.loads(args.out.read_text()) if args.out.exists() else {}
    block = numbers.setdefault(args.agent, {}).setdefault(args.benchmark, {}).setdefault(args.regime, {})
    block.setdefault("p2", {})["kappa_miss"] = result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
