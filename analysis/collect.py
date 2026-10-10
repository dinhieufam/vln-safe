#!/usr/bin/env python3
"""Walk results/, write results/summary/paper_numbers.json.

What: for every (agent, benchmark, regime) with a landed results/<agent>/<benchmark>/
<regime>/episodes.csv, calls analysis/profile.py's arm_profile() and records that arm's
joint outcome-by-contact counts/rates (analysis/make_figures.py's fig_autopsy reads these),
P1 and P2 (proxy-vs-engine validity on that arm's own episodes), plus a provenance sub-block
(episode count, split, seed, checkpoint, a subset flag, this repo's git sha). Where both
regimes exist for an (agent, benchmark) pair it also writes `regime_delta`: the P2 regime
quantities the paper reports -- delta-SR and delta-dirty-success from block to slide, and
kappa-hat-miss, read from the *slide* arm's P2 (slide is the regime the displacement proxy
actually fails in: a grazing collision keeps the agent moving, so the proxy sees ordinary
forward progress; under block a blocked collision looks the same as a blocked non-collision
either way, so kappa-hat-miss there is uninformative.

Guarded arms (the method, `<regime>_guard-<name>/`) are not walked here: their paired
comparison needs the step logs, so analysis/guard_effect.py writes
results/summary/guard_effect.json and this script copies each entry under
`<agent>.<benchmark>.guard.<regime>.<name>` -- the same file therefore holds every number the
paper quotes. Run guard_effect.py first when a guarded arm lands.

No table is generated here -- analysis/make_tables.py reads this file. Nothing is
transcribed by hand in either direction: an arm that has not landed is simply absent from
the JSON, and make_tables.py renders that absence as `\\TBD`.

Usage: analysis/collect.py [--out results/summary/paper_numbers.json]
Modern Python (3.9+, pandas); never imported by the harness.
"""
import argparse
import glob
import importlib.util
import json
import os
import subprocess

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

_spec = importlib.util.spec_from_file_location("_vlnsafe_analysis_profile", os.path.join(HERE, "profile.py"))
_profile = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_profile)

BENCHMARKS = ("r2r", "rxr")
REGIMES = ("block", "slide")
# Unguarded comparison conditions that are not a regime of their own, collected with the
# same profile under `<agent>.<benchmark>.baseline.<name>`: ETPNav-lineage's own Tryout
# collision-escape heuristic, switched on.
BASELINE_ARMS = ("block_tryout",)


def _agent_ids():
    paths = glob.glob(os.path.join(REPO, "configs", "agents", "*.yaml"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _load_manifest(arm_dir):
    path = os.path.join(arm_dir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _is_subset(manifest):
    """True iff this arm's episodes are a strict subset of the official split -- the
    condition that puts a dagger on the number in tab_backbone/tab_regime. A manifest that says `full_split: true` is authoritative; absent that,
    a non-null `episode_subset` (the pre-Phase-B migrated-archive field) means subset."""
    if manifest.get("full_split") is True:
        return False
    subset = manifest.get("episode_subset")
    return subset not in (None, {})


EPISODE_MEANS = ("success", "spl", "ndtw", "navigation_error_m", "collision_count",
                 "collision_event_count", "stuck_flag", "step_count")


def episode_means(df):
    """Per-episode means of the task and contact columns, so an unguarded arm (or a
    baseline) can sit in the same table row format as a guarded one."""
    out = {}
    for col in EPISODE_MEANS:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            out[col] = float(v.mean()) if v.notna().any() else None
    return out


def collect_arm(agent, benchmark, regime, git_sha):
    arm_dir = os.path.join(REPO, "results", agent, benchmark, regime)
    ep_csv = os.path.join(arm_dir, "episodes.csv")
    if not os.path.exists(ep_csv):
        return None
    df = pd.read_csv(ep_csv).drop_duplicates(subset=["episode_id"], keep="first")
    profile = _profile.arm_profile(df)
    manifest = _load_manifest(arm_dir)
    return {
        "episode_means": episode_means(df),
        "n_episodes": profile["n_episodes"],
        "n_success": profile["n_success"],
        "joint_counts": profile["joint_counts"],
        "joint_rates": profile["joint_rates"],
        "p1": profile["p1"],
        "p2": profile["p2"],
        "provenance": {
            "episodes_csv": os.path.relpath(ep_csv, REPO),
            "episode_count": profile["n_episodes"],
            "split": manifest.get("split"),
            "seed": manifest.get("seed"),
            "checkpoint": manifest.get("checkpoint"),
            "subset": _is_subset(manifest),
            "git_sha": git_sha,
        },
    }


def regime_delta(block, slide):
    """P2's regime quantities: computed only when both regimes have landed for this
    (agent, benchmark).

    Delta-SR and delta-dirty-success are the difference of the two arms' headline numbers
    **as displayed** (each rounded to 1 decimal place, then subtracted), not of the exact
    floats. Gate E caught this: diff-then-round instead gives ETPNav R2R +15.3/+8.3pp and
    RxR dirty +15.3pp against the hand-written table's +15.4/+8.4/+15.4 -- a silent 0.1pp
    drift that would otherwise make Table 3 and Table 6 inconsistent with each other (a
    reader subtracting Table 3's own printed numbers would get a different answer than
    Table 6 shows). Round-then-diff keeps every table self-consistent under hand
    arithmetic, and reproduces today's typed numbers exactly."""
    if block is None or slide is None:
        return None
    sr_block = round(100.0 * block["n_success"] / block["n_episodes"], 1) if block["n_episodes"] else float("nan")
    sr_slide = round(100.0 * slide["n_success"] / slide["n_episodes"], 1) if slide["n_episodes"] else float("nan")
    dirty_block = round(100.0 * block["p1"]["dirty_success_rate"], 1)
    dirty_slide = round(100.0 * slide["p1"]["dirty_success_rate"], 1)
    return {
        "delta_sr_pp": round(sr_slide - sr_block, 1),
        "delta_dirty_success_pp": round(dirty_slide - dirty_block, 1),
        "kappa_miss": slide["p2"].get("kappa_miss"),
        "kappa_miss_source_regime": "slide",
    }


def attach_guard_arms(out, path=None):
    """Copy results/summary/guard_effect.json's entries (keyed `agent/bench/regime_guard-name`)
    into out[agent][bench]["guard"][regime][name]. A missing file leaves out unchanged."""
    path = path or os.path.join(REPO, "results", "summary", "guard_effect.json")
    if not os.path.exists(path):
        return out
    with open(path) as f:
        guard = json.load(f)
    for key, entry in guard.items():
        agent, benchmark, arm = key.split("/")
        regime, name = arm.split("_guard-", 1)
        (out.setdefault(agent, {}).setdefault(benchmark, {}).setdefault("guard", {})
            .setdefault(regime, {}))[name] = entry
    return out


def attach_baseline_pairs(out, path=None):
    """Copy results/summary/baseline_effect.json (analysis/guard_effect.py; keyed
    `agent/bench/<baseline arm>`) into out[agent][bench]["baseline"][arm]["paired"]."""
    path = path or os.path.join(REPO, "results", "summary", "baseline_effect.json")
    if not os.path.exists(path):
        return out
    with open(path) as f:
        pairs = json.load(f)
    for key, entry in pairs.items():
        agent, benchmark, arm = key.split("/")
        base = out.get(agent, {}).get(benchmark, {}).get("baseline", {}).get(arm)
        if base is not None:
            base["paired"] = entry
    return out


def collect():
    sha = _git_sha()
    out = {}
    for agent in _agent_ids():
        agent_block = {}
        for benchmark in BENCHMARKS:
            arms = {}
            for regime in REGIMES:
                arm = collect_arm(agent, benchmark, regime, sha)
                if arm is not None:
                    arms[regime] = arm
            baselines = {b: collect_arm(agent, benchmark, b, sha) for b in BASELINE_ARMS}
            baselines = {b: v for b, v in baselines.items() if v is not None}
            if not arms and not baselines:
                continue
            bench_block = dict(arms)
            if baselines:
                bench_block["baseline"] = baselines
            delta = regime_delta(arms.get("block"), arms.get("slide"))
            if delta is not None:
                bench_block["regime_delta"] = delta
            agent_block[benchmark] = bench_block
        if agent_block:
            out[agent] = agent_block
    return attach_baseline_pairs(attach_guard_arms(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "summary", "paper_numbers.json"))
    args = ap.parse_args()

    out = collect()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    n_arms = sum(1 for a in out.values() for b in a.values() for r in REGIMES if r in b)
    print("wrote {} ({} agents, {} arms)".format(args.out, len(out), n_arms))
    for agent in sorted(out):
        for benchmark in sorted(out[agent]):
            landed = [r for r in REGIMES if r in out[agent][benchmark]]
            guarded = ["{}_guard-{}".format(r, n)
                       for r, names in out[agent][benchmark].get("guard", {}).items() for n in names]
            print("  {}/{}: {}".format(agent, benchmark, ", ".join(landed + sorted(guarded))))


if __name__ == "__main__":
    main()
