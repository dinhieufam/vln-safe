#!/usr/bin/env python3
"""Guard-off regression gate (the guard layer must be a no-op when off): a run made with the current harness and no --guard must reproduce the
landed arm cell for cell.

Compares every column the two schemas share, as the literal CSV strings, on
episodes.csv (matched on episode_id) and steps.csv (matched on episode_id, step_index),
for the episodes the new run contains. Columns only the new schema has (guard_*) must be
UNAVAILABLE on every row. Reading the landed side accepts .csv or .csv.gz.

The gate run must replay the landed episodes in the landed order from a fresh process,
e.g. the first 25 episodes of shard 0 of a 2-shard arm:
  python -m vlnsafe run --agent etpr1 --benchmark r2r --regime block \\
      --episodes 50 --shards 2 --shard-index 0 --out $VLNSAFE_SCRATCH/n9_gate/guard_off

Usage:
  scripts/check_guard_off.py NEW_DIR LANDED_DIR [--shard-suffix .shard0]
Exit 0 if zero cells differ, 1 otherwise.
"""
import argparse
import csv
import gzip
import os
import sys

UNAVAILABLE = "UNAVAILABLE"


def _open(path):
    if os.path.exists(path):
        return open(path, newline="")
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt", newline="")
    raise SystemExit("missing: {} (or .gz)".format(path))


def _rows(path, key, keep=None):
    with _open(path) as f:
        reader = csv.DictReader(f)
        out = {}
        for row in reader:
            if keep is not None and row["episode_id"] not in keep:
                continue
            out[tuple(row[k] for k in key)] = row
        return reader.fieldnames, out


def _compare(label, new_path, old_path, key, keep=None, skip=()):
    new_cols, new = _rows(new_path, key, keep)
    old_cols, old = _rows(old_path, key, set(r[0] for r in new) if keep is None else keep)
    shared = [c for c in new_cols if c in old_cols and c not in skip]
    extra = [c for c in new_cols if c not in old_cols]
    missing_rows = sorted(set(new) ^ set(old))
    diffs = {}
    for k in sorted(set(new) & set(old)):
        for c in shared:
            if new[k][c] != old[k][c]:
                diffs.setdefault(c, []).append((k, new[k][c], old[k][c]))
    not_unavailable = sum(1 for r in new.values() for c in extra if r[c] != UNAVAILABLE)
    cells = len(set(new) & set(old)) * len(shared)
    print("{}: {} rows, {} shared columns, {} cells compared, {} differ; "
          "{} rows on one side only; new-only columns {} ({} non-UNAVAILABLE cells)".format(
              label, len(new), len(shared), cells, sum(len(v) for v in diffs.values()),
              len(missing_rows), extra, not_unavailable))
    for c, items in sorted(diffs.items()):
        print("  {}: {} diffs, first {}".format(c, len(items), items[0]))
    return not diffs and not missing_rows and not not_unavailable, set(r[0] for r in new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new_dir")
    ap.add_argument("landed_dir")
    ap.add_argument("--shard-suffix", default="",
                    help="suffix of the NEW run's files, e.g. .shard0 (landed side is merged)")
    ap.add_argument("--skip-column", action="append", default=[],
                    help="shared column not compared, e.g. predicted_clearance_m for a T1 "
                         "arm landed before N3 (UNAVAILABLE there, live now)")
    args = ap.parse_args()
    s = args.shard_suffix
    ok_ep, episodes = _compare(
        "episodes", os.path.join(args.new_dir, "episodes{}.csv".format(s)),
        os.path.join(args.landed_dir, "episodes.csv"), ("episode_id",),
        skip=args.skip_column)
    ok_st, _ = _compare(
        "steps", os.path.join(args.new_dir, "steps{}.csv".format(s)),
        os.path.join(args.landed_dir, "steps.csv"), ("episode_id", "step_index"),
        keep=episodes, skip=args.skip_column)
    if args.skip_column:
        print("not compared: {}".format(args.skip_column))
    ok = ok_ep and ok_st
    print("GATE {}".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
