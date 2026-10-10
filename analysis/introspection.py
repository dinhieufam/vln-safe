#!/usr/bin/env python3
"""P3 -- does a signal the agent (or the harness) already has predict a collision before
it happens? AUROC/AP/precision-at-recall for each decision-time signal against a
k-primitive-step-ahead engine-collision label, per (agent, benchmark), under the `block`
regime (the stricter regime, and the one the archived comparator data was computed under
).

Five signals, unified across tiers so one scoring loop covers both:
  - `entropy`   : T2's own `candidate_entropy` where the worker exposes a spatial
                  candidate set, else the Shannon entropy of a T1 agent's 4-way
                  {stop, forward, left, right} categorical -- same "how spread out is the
                  probability mass" question, computed on whichever candidate set exists.
  - `margin_inv`: T2's own `top2_margin`, else the gap between a T1 categorical's top two
                  probabilities. Inverted (low margin -> high risk score).
  - `stop_prob_inv`: T2's `stop_prob`, else T1's `prob_stop` directly -- both are "the
                  probability mass the agent put on stopping." Inverted.
  - `entropy_norm`: T2's `candidate_entropy_norm` only (no T1 analogue: normalisation is
                  against the T2 candidate set's own size, which T1's fixed 4-way
                  categorical does not need).
  - `stall_history`: 1 if the agent stands at a pose (position, heading) from which an
                  executed forward already moved < STALL_M this episode, else 0 -- the
                  guard's stall veto (vlnsafe/guard.py, `stall_m`), recomputed offline from
                  odometry alone. Pose is rebuilt from the log: position changes only on a
                  forward that moved >= STALL_M, heading is the net turn count modulo the
                  benchmark's turns per revolution (read from its config). Also scored per
                  executed forward step (`stall_history_forward`): precision, recall, and
                  the share of collision steps it covers.
  - `geometry_inv`: the harness's own depth-projected clearance prediction,
                  `predicted_clearance_m`, inverted. T2 (ETPNav-lineage worker) logs it
                  live. The T1 arms landed before the predictor was wired for them, so
                  their column is recovered by scripts/replay_geometry.py (a deterministic
                  re-simulation, no model) and joined here from
                  data/scratch/<arm>/geometry_replay/shard*/geometry.csv.gz; an arm with no
                  sidecar stays UNAVAILABLE.

`signals_first_contact` scores the same signals on the decision steps
where contact would be *new*: no earlier stall at this pose (stall_history = 0, so the
history signal is constant there by construction and has no AUROC) and no collision on the
previous primitive. The full-set history AUROC measures repeat contact; this subset asks
which source warns before the first one.

Usage: analysis/introspection.py [--out results/summary/introspection.json]
Modern Python (3.9+, pandas, numpy); never imported by the harness.

Ported from the pre-release analyze_e5_introspective_risk.py (per-agent
RUNS dict, ETPNav only) generalised to walk every landed `block`-regime arm and to unify
T1/T2 signal columns instead of scoring T2's `internal_*` columns alone.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

BENCHMARKS = ("r2r", "rxr")
REGIME = "block"           # P3/P4 have always been computed under block
HORIZONS = (1, 3, 5)

_NUMERIC_COLS = [
    "step_index", "is_decision", "engine_collision_flag",
    "prob_stop", "prob_forward", "prob_left", "prob_right",
    "candidate_entropy", "candidate_entropy_norm", "top2_margin", "stop_prob",
    "num_candidates", "value_estimate", "predicted_clearance_m",
    "min_eps_for_safe_alt", "safe_alt_exists_eps10",
]

# signal name -> (unified column, sign). sign=+1: higher raw value = higher risk;
# sign=-1: lower raw value = higher risk, so the scored value is -1 * column.
SIGNALS = {
    "entropy": ("entropy_unified", 1.0),
    "entropy_norm": ("candidate_entropy_norm", 1.0),
    "margin_inv": ("margin_unified", -1.0),
    "stop_prob_inv": ("stop_prob_unified", -1.0),
    "geometry_inv": ("predicted_clearance_m", -1.0),
    "stall_history": ("stall_history", 1.0),
}
STALL_M = 0.01             # configs/guard/stall-mem1.yaml `stall_m`


def _agent_ids():
    paths = glob.glob(os.path.join(REPO, "configs", "agents", "*.yaml"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


def attach_replay_geometry(df, sidecar_paths):
    """Fill `predicted_clearance_m` from scripts/replay_geometry.py sidecars
    (episode_id, step_index, predicted_clearance_m), joined on (episode_id, step_index).
    Only touches a column with no data at all -- an arm whose own run already logged the
    value (T2) is never overwritten. Rows of an episode the replay could not reproduce
    stay NaN (UNAVAILABLE), never zero. Returns (df, n_rows_filled)."""
    if not sidecar_paths or df["predicted_clearance_m"].notna().any():
        return df, 0
    parts = [pd.read_csv(p, dtype={"episode_id": str}) for p in sidecar_paths]
    geo = pd.concat(parts, ignore_index=True)
    geo["predicted_clearance_m"] = pd.to_numeric(geo["predicted_clearance_m"], errors="coerce")
    geo = geo.drop_duplicates(["episode_id", "step_index"])
    # join on a string key so df keeps its own episode_id dtype (and therefore its sort
    # order, which the tie-sensitive AP / precision-at-recall numbers depend on)
    geo = geo.assign(_eid=geo["episode_id"].astype(str)).drop(columns=["episode_id"])
    out = df.assign(_eid=df["episode_id"].astype(str)).drop(columns=["predicted_clearance_m"])
    out = out.merge(geo, on=["_eid", "step_index"], how="left").drop(columns=["_eid"])
    return out, int(out["predicted_clearance_m"].notna().sum())


def _replay_sidecars(agent, benchmark):
    root = os.path.join(REPO, "data", "scratch", agent, benchmark, REGIME, "geometry_replay")
    return sorted(glob.glob(os.path.join(root, "shard*", "geometry.csv.gz")))


def _load_steps(agent, benchmark):
    path = os.path.join(REPO, "results", agent, benchmark, REGIME, "steps.csv.gz")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df, n_filled = attach_replay_geometry(df, _replay_sidecars(agent, benchmark))
    if n_filled:
        print("  {}/{}: geometry column filled from replay for {} of {} rows".format(
            agent, benchmark, n_filled, len(df)))
    # A resumed episode's stale partial rows precede its complete re-run in the file
    df = df.drop_duplicates(["episode_id", "step_index"], keep="last")
    return df.sort_values(["episode_id", "step_index"]).reset_index(drop=True)


def add_unified_signals(df):
    """entropy_unified/margin_unified/stop_prob_unified: T2's own column where present,
    else derived from a T1 agent's {stop, forward, left, right} categorical -- the same
    quantity (spread of the action distribution, gap between its top two, mass on
    stopping), read off whichever candidate set the agent exposes."""
    df = df.copy()
    probs = df[["prob_stop", "prob_forward", "prob_left", "prob_right"]]
    has_categorical = probs.notna().all(axis=1)
    cat_entropy = pd.Series(np.nan, index=df.index)
    cat_margin = pd.Series(np.nan, index=df.index)
    if has_categorical.any():
        p = np.clip(probs.loc[has_categorical].to_numpy(), 1e-12, 1.0)
        cat_entropy.loc[has_categorical] = -(p * np.log(p)).sum(axis=1)
        top2 = np.sort(p, axis=1)[:, -2:]
        cat_margin.loc[has_categorical] = top2[:, 1] - top2[:, 0]
    df["entropy_unified"] = df["candidate_entropy"].combine_first(cat_entropy)
    df["margin_unified"] = df["top2_margin"].combine_first(cat_margin)
    df["stop_prob_unified"] = df["stop_prob"].combine_first(df["prob_stop"])
    return df


def _turns_per_revolution(benchmark):
    import yaml
    with open(os.path.join(REPO, "configs", "benchmarks", "{}.yaml".format(benchmark))) as f:
        turn = float(yaml.safe_load(f)["rig"]["turn_angle_deg"])
    return int(round(360.0 / turn))


def add_stall_history(df, benchmark, stall_m=STALL_M):
    """`stall_history` per step (see the module docstring). Requires rows sorted by
    (episode_id, step_index); duplicate (episode_id, step_index) rows must already
    be dropped."""
    n_rev = _turns_per_revolution(benchmark)
    eids = df["episode_id"].to_numpy()
    acts = df["executed_action"].astype(str).to_numpy()
    disp = pd.to_numeric(df["sim_displacement_m"], errors="coerce").fillna(0.0).to_numpy()
    out = np.zeros(len(df), dtype=np.int8)
    prev_eid, pos, head, blocked = None, 0, 0, set()
    for i in range(len(df)):
        if eids[i] != prev_eid:
            prev_eid, pos, head, blocked = eids[i], 0, 0, set()
        out[i] = 1 if (pos, head) in blocked else 0
        a = acts[i]
        if a == "move_forward":
            if disp[i] < stall_m:
                blocked.add((pos, head))
            else:
                pos += 1
        elif a == "turn_left":
            head = (head + 1) % n_rev
        elif a == "turn_right":
            head = (head - 1) % n_rev
    df = df.copy()
    df["stall_history"] = out
    return df


def score_stall_forward(df):
    """The stall veto on its own domain: executed forward steps. Precision = P(collide |
    forward from a blocked pose); recall = share of collided forwards it would veto."""
    fwd = df[df["executed_action"].astype(str) == "move_forward"]
    y = fwd["engine_collision_flag"].fillna(0).astype(int)
    s = fwd["stall_history"]
    flagged = int(s.sum())
    tp = int(((s == 1) & (y == 1)).sum())
    n_pos = int(y.sum())
    return {
        "n_forward_steps": int(len(fwd)),
        "n_collided_forward_steps": n_pos,
        "n_flagged": flagged,
        "n_true_positive": tp,
        "precision": tp / flagged if flagged else None,
        "recall": tp / n_pos if n_pos else None,
        "auroc": roc_auc(y, s)[0],
    }


def add_horizon_labels(df, horizons=HORIZONS):
    """label_k[i] = 1 if engine_collision_flag==1 at any step in [step_index_i,
    step_index_i + k - 1] within the same episode (the decision at step i governs the
    primitive about to execute at step i itself, so the window is inclusive of i).
    Equivalent to the old row-by-row `coll[i:i+k].max()`, computed as a reversed rolling
    max so it stays O(n) instead of O(n*k) per episode."""
    df = df.copy()
    coll = df["engine_collision_flag"].fillna(0)
    for k in horizons:
        label = coll.groupby(df["episode_id"]).transform(
            lambda s: s[::-1].rolling(k, min_periods=1).max()[::-1]
        )
        df["label_k{}".format(k)] = label.astype(int)
    return df


def roc_auc(y, scores):
    """Mann-Whitney U / AUC identity: AUC = (sum of average-tie ranks among the positives
    - n_pos*(n_pos+1)/2) / (n_pos*n_neg). Exactly the pairwise "P(pos score > neg score),
    ties worth half" definition (verified against a brute-force pairwise implementation
    across randomized trials with injected ties, max abs diff < 1e-9) but O(n log n)
    instead of O(n_pos * n_neg) -- the pairwise loop took minutes on RxR's ~26k decision
    steps at a ~60% positive rate."""
    sub = pd.DataFrame({"y": y, "s": scores}).dropna()
    if len(sub) < 3 or sub["y"].nunique() < 2:
        return None, int(len(sub))
    y = sub["y"].to_numpy()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, int(len(sub))
    ranks = sub["s"].rank(method="average").to_numpy()
    sum_ranks_pos = float(ranks[y == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), int(len(sub))


def average_precision(y, scores):
    sub = pd.DataFrame({"y": y, "s": scores}).dropna().sort_values("s", ascending=False)
    if len(sub) < 3 or sub["y"].nunique() < 2:
        return None
    y = sub["y"].to_numpy()
    n_pos = float(y.sum())
    if n_pos == 0:
        return None
    tp, ap = 0.0, 0.0
    for i, label in enumerate(y, start=1):
        if label == 1:
            tp += 1.0
            ap += tp / i
    return float(ap / n_pos)


def precision_at_recall(y, scores, target_recall=0.9):
    sub = pd.DataFrame({"y": y, "s": scores}).dropna().sort_values("s", ascending=False)
    if len(sub) < 3 or sub["y"].nunique() < 2:
        return None
    y = sub["y"].to_numpy()
    n_pos = float(y.sum())
    tp, fp, best = 0.0, 0.0, None
    for label in y:
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        recall = tp / n_pos
        if recall >= target_recall:
            precision = tp / (tp + fp)
            best = precision if best is None else max(best, precision)
    return best


def score_signals(dec):
    out = {}
    for name, (col, sign) in SIGNALS.items():
        if col not in dec.columns:
            continue
        per_k = {}
        for k in HORIZONS:
            y = dec["label_k{}".format(k)]
            s = sign * dec[col]
            auc, n = roc_auc(y, s)
            per_k[k] = {
                "n": n,
                "positive_rate": float(y.mean()) if len(y) else None,
                "auroc": auc,
                "average_precision": average_precision(y, s),
                "precision_at_recall_0.9": precision_at_recall(y, s, 0.9),
            }
        out[name] = per_k
    return out


def analyze_arm(agent, benchmark):
    df = _load_steps(agent, benchmark)
    if df is None:
        return None
    df = add_unified_signals(df)
    df = add_stall_history(df, benchmark)
    df = add_horizon_labels(df)
    prev_coll = df.groupby("episode_id")["engine_collision_flag"].shift(1).fillna(0)
    dec = df[df["is_decision"] == 1]
    first = df[(df["is_decision"] == 1) & (df["stall_history"] == 0) & (prev_coll == 0)]
    return {
        "n_first_contact_decision_steps": int(len(first)),
        "signals_first_contact": score_signals(first),
        "regime": REGIME,
        "n_rows": int(len(df)),
        "n_decision_steps": int(len(dec)),
        "n_episodes": int(df["episode_id"].nunique()),
        "collision_step_rate": float(df["engine_collision_flag"].mean()),
        "signals_at_decision_steps": score_signals(dec),
        "stall_history_forward": score_stall_forward(df),
    }


def collect():
    out = {}
    for agent in _agent_ids():
        agent_block = {}
        for benchmark in BENCHMARKS:
            arm = analyze_arm(agent, benchmark)
            if arm is not None:
                agent_block[benchmark] = arm
        if agent_block:
            out[agent] = agent_block
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "summary", "introspection.json"))
    args = ap.parse_args()

    out = collect()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print("wrote {} ({} agents)".format(args.out, len(out)))
    for agent in sorted(out):
        print("  {}: {}".format(agent, ", ".join(sorted(out[agent]))))


if __name__ == "__main__":
    main()
