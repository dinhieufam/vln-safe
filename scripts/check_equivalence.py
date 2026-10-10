#!/usr/bin/env python3
"""Equivalence gate: a freshly produced episodes.csv vs the
migrated per-episode rows, column by column, matched on episode_id.

The migrated files under results/ (results/provenance.md) already use the new schema: both sides use `collision_count`
for the physics-engine predicate, `navigation_error_m`, `path_length_m`, etc. There is no
column renaming left to do here; a prior version of this mapping still pointed at the
pre-migration names (`navigation_error`, `native_collision_count`, `path_length`),
which do not exist in the migrated archive and made every comparison on those three
columns fail regardless of actual equivalence.

Exact-match columns: success, spl, navigation_error_m, collision_count, nc_rate,
step_count, path_length_m. Near-match (1e-6): min_clearance_m.

Usage:
  scripts/check_equivalence.py NEW_episodes.csv ARCHIVED_per_episode.csv [--episodes N]
Exit 0 if every compared cell matches, 1 otherwise.
"""
import argparse
import csv
import sys

# new column -> archived column (both already use the new schema; identity mapping)
EXACT = {
    "success": "success",
    "spl": "spl",
    "navigation_error_m": "navigation_error_m",
    "collision_count": "collision_count",   # physics-engine predicate in both
    "nc_rate": "nc_rate",
    "step_count": "step_count",
    "path_length_m": "path_length_m",
}
NEAR = {"min_clearance_m": "min_clearance_m"}
NEAR_TOL = 1e-6
EXACT_TOL = 1e-9  # float reprs, not bit patterns


def _read(path):
    with open(path, newline="") as f:
        return {row["episode_id"]: row for row in csv.DictReader(f)}


def _f(x):
    return None if x in ("", "UNAVAILABLE", None) else float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_csv")
    ap.add_argument("archived_csv")
    ap.add_argument("--episodes", type=int, default=None)
    args = ap.parse_args()

    new, old = _read(args.new_csv), _read(args.archived_csv)
    shared = [e for e in new if e in old]
    if args.episodes:
        shared = shared[: args.episodes]
    if not shared:
        print("no shared episode_ids between the two files", file=sys.stderr)
        return 1

    failures = []
    for eid in shared:
        for new_col, old_col in EXACT.items():
            a, b = _f(new[eid].get(new_col)), _f(old[eid].get(old_col))
            if a is None or b is None:
                if a != b:
                    failures.append((eid, new_col, a, b))
            elif abs(a - b) > EXACT_TOL + 1e-9 * abs(b):
                failures.append((eid, new_col, a, b))
        for new_col, old_col in NEAR.items():
            a, b = _f(new[eid].get(new_col)), _f(old[eid].get(old_col))
            if a is None or b is None:
                if a != b:
                    failures.append((eid, new_col, a, b))
            elif abs(a - b) > NEAR_TOL:
                failures.append((eid, new_col, a, b))

    print("compared {} episodes x {} columns".format(shared and len(shared), len(EXACT) + len(NEAR)))
    if failures:
        for eid, col, a, b in failures[:40]:
            print("  MISMATCH ep {} {}: new={} archived={}".format(eid, col, a, b))
        print("-- {} mismatched cells --".format(len(failures)))
        return 1
    print("-- all cells match --")
    return 0


if __name__ == "__main__":
    sys.exit(main())
