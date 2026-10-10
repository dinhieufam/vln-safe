#!/usr/bin/env python3
"""Concatenate an arm's shard outputs into episodes.csv + steps.csv.gz, dedupe on
episode_id, write a merged manifest.

Usage: scripts/merge_shards.py results/<agent>/<benchmark>/<regime>
Reads episodes.shard*.csv / steps.shard*.csv in that dir (or the single unsharded pair),
gzips the merged step log into $VLNSAFE_SCRATCH and leaves a symlink beside episodes.csv.
"""
import argparse
import csv
import glob
import gzip
import json
import os
import shutil
import sys


def _merge_csv(paths, out_path, key="episode_id"):
    seen, rows, header = set(), [], None
    for p in sorted(paths):
        with open(p, newline="") as f:
            reader = csv.reader(f)
            h = next(reader)
            header = header or h
            if h != header:
                raise SystemExit("shard {} has a different header".format(p))
            ki = header.index(key) if key in header else None
            for row in reader:
                if ki is not None:
                    if row[ki] in seen:
                        continue
                    seen.add(row[ki])
                rows.append(row)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    args = ap.parse_args()
    d = args.arm_dir

    ep_shards = glob.glob(os.path.join(d, "episodes.shard*.csv")) or \
        [os.path.join(d, "episodes.csv")]
    st_shards = glob.glob(os.path.join(d, "steps.shard*.csv")) or \
        [os.path.join(d, "steps.csv")]
    ep_shards = [p for p in ep_shards if os.path.exists(p)]
    st_shards = [p for p in st_shards if os.path.exists(p)]
    if not ep_shards:
        raise SystemExit("no episodes*.csv in {}".format(d))

    n_ep = _merge_csv(ep_shards, os.path.join(d, "episodes.csv"))

    scratch = os.path.abspath(os.environ.get("VLNSAFE_SCRATCH") or os.path.join(
        os.path.dirname(__file__), "..", "data", "scratch"))
    rel = os.path.relpath(os.path.abspath(d),
                          os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")))
    step_dir = os.path.join(scratch, rel)
    os.makedirs(step_dir, exist_ok=True)
    merged_steps = os.path.join(step_dir, "steps.csv")
    n_st = _merge_csv(st_shards, merged_steps, key=None) if st_shards else 0
    gz = merged_steps + ".gz"
    with open(merged_steps, "rb") as fi, gzip.open(gz, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.remove(merged_steps)
    link = os.path.join(d, "steps.csv.gz")
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link)
    # gz must be absolute: this symlink lives in results/<arm>/, not in the directory the
    # script was invoked from, so a relative target (scripts/../data/scratch/...) resolves
    # against the wrong base and silently dangles (found 2026-09-19, all four ETP-R1 arms).
    os.symlink(gz, link)

    # prefer a shard's own manifest (this run's fresh metadata) over a pre-existing
    # manifest.json -- that name is also this function's own output path, so a stale one
    # left by an earlier/different run (e.g. a migrated subset) would otherwise sort first
    # ("manifest.json" < "manifest.shard0.json") and its now-wrong fields (episode_subset,
    # full_split, checkpoint, ...) would silently survive into the merged result.
    manifests = sorted(glob.glob(os.path.join(d, "manifest.shard*.json"))) or \
        sorted(glob.glob(os.path.join(d, "manifest*.json")))
    if manifests:
        merged = json.load(open(manifests[0]))
        merged["episode_count"] = n_ep
        merged["merged_from"] = [os.path.basename(p) for p in sorted(manifests)]
        json.dump(merged, open(os.path.join(d, "manifest.json"), "w"), indent=2, sort_keys=True)

    print("merged {} episode rows, {} step rows -> {}".format(n_ep, n_st, d))


if __name__ == "__main__":
    sys.exit(main())
