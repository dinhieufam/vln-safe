#!/usr/bin/env python3
"""P6 -- Phase 2b: three signal-completeness decisions for
contribution 2, decided and recorded here rather than left as a `\\TBD` gap.

1. OBSTACLE-REGION ATTENTION -- DROPPED from contribution 2's claim.
   Investigated `external/ETPNav/vlnce_baselines/models/etp/vilmodel_cmt.py`: the only
   cross-attention on the path to `global_logits` (`forward_navigation`, ~line 738,
   `self.global_encoder.encoder(txt_embeds, txt_masks, gmap_embeds, ...)`) is text-to-
   candidate-node attention over `gmap_embeds` -- the same per-candidate embedding that
   already backs `candidate_entropy` / `top2_margin_prob` / `stop_prob`
   (`vlnsafe/workers/etpnav_worker.py:281-290`), the internal signals section 2's AUROC
   table already measures near chance (.48-.69, five agents). It is not a per-pixel or
   per-direction spatial signal: `forward_panorama`'s `pano_encoder` (line 715-718) pools
   view features into node embeddings *before* the graph is built, and is not
   obstacle-conditioned. Extracting and reporting it would either restate an
   already-reported null result under a new name, or require building a probe the
   architecture does not expose. It is also structurally unavailable for 3 of 5 headline
   agents (CMA / Seq2Seq / StreamVLN have no candidate graph), so even a positive result
   would not extend the geometry claim's five-agent uniformity. Contribution 2 reports
   obstacle-region attention as "not available in a form independent of already-measured
   signals," not silently dropped.

   Noted for the record, not acted on: CMA / Seq2Seq (`external/VLN-CE/vlnce_baselines/
   models/cma_policy.py:207-217`, called at line 274) *do* compute a genuine spatial
   attention over the flattened depth-CNN feature grid (`depth_embedding =
   self._attn(text_q, depth_k, depth_v)`) -- closer to a literal obstacle-region signal
   than anything in ETPNav. Out of scope here (this analysis targets the
   ETPNav-lineage signal specifically); flagged as a possible follow-up only if a
   reviewer pushes on the attention claim, not committed to now.

2. VALUE / PROGRESS -- reported as structurally absent, not measured as chance.
   `vlnsafe/rollout.py:169` already logs `value_estimate` as `UNAVAILABLE`
   unconditionally ("GRPO/ETPNav: no value head") -- a coding decision made before this
   analysis, now the recorded claim rather than a TBD. ETP-R1 is GRPO (critic-free,
   verified against the released state dict: no value-head parameter names in the
   checkpoint); ETPNav has no value head in its released architecture; CMA / Seq2Seq
   (habitat-lab baselines) optionally support a *progress* monitor
   (`external/VLN-CE/vlnce_baselines/models/cma_policy.py:201-205`,
   `PROGRESS_MONITOR.use`) but the released checkpoints used here were not trained with
   it enabled, so the head does not exist in the checkpoint either. Contribution 2 reports
   this as a structural property of the released model zoo, not a per-agent gap.

3. COMBINATION (internal + external) -- a two-feature logistic fit, in-sample, reported
   as an upper bound only; the method itself stays training-free. Features: `entropy_unified` (the one internal signal with a
   genuinely above-chance AUROC anywhere, ETP-R1 .69 -- section 2) and `geometry_score`
   (-predicted_clearance_m), standardized; fit by Newton-Raphson / IRLS, not
   scikit-learn (this repo has no scikit-learn dependency -- the same choice
   `introspection.py::roc_auc` already made for AUROC). Label: `label_k1` (collision at
   the decision step itself, same as headroom.py's risky-decision definition).

Usage: analysis/signal_completeness.py [--out results/summary/signal_completeness.json]
Modern Python (3.9+, pandas, numpy); never imported by the harness.
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
REGIME = "block"

_NUMERIC_COLS = [
    "step_index", "is_decision", "engine_collision_flag",
    "prob_stop", "prob_forward", "prob_left", "prob_right",
    "candidate_entropy", "predicted_clearance_m",
]


def _agent_ids():
    paths = glob.glob(os.path.join(REPO, "configs", "agents", "*.yaml"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


def attach_replay_geometry(df, sidecar_paths):
    """Same join as analysis/introspection.py::attach_replay_geometry; duplicated
    rather than imported, matching analysis/predictability.py's own precedent."""
    if not sidecar_paths or df["predicted_clearance_m"].notna().any():
        return df, 0
    parts = [pd.read_csv(p, dtype={"episode_id": str}) for p in sidecar_paths]
    geo = pd.concat(parts, ignore_index=True)
    geo["predicted_clearance_m"] = pd.to_numeric(geo["predicted_clearance_m"], errors="coerce")
    geo = geo.drop_duplicates(["episode_id", "step_index"])
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
    df, _ = attach_replay_geometry(df, _replay_sidecars(agent, benchmark))
    return df.sort_values(["episode_id", "step_index"]).reset_index(drop=True)


def add_entropy_unified(df):
    """Same derivation as introspection.py::add_unified_signals's entropy_unified (T2's
    own `candidate_entropy` where present, else the Shannon entropy of a T1 agent's
    4-way categorical); duplicated for the same reason as attach_replay_geometry above."""
    df = df.copy()
    probs = df[["prob_stop", "prob_forward", "prob_left", "prob_right"]]
    has_categorical = probs.notna().all(axis=1)
    cat_entropy = pd.Series(np.nan, index=df.index)
    if has_categorical.any():
        p = np.clip(probs.loc[has_categorical].to_numpy(), 1e-12, 1.0)
        cat_entropy.loc[has_categorical] = -(p * np.log(p)).sum(axis=1)
    df["entropy_unified"] = df["candidate_entropy"].combine_first(cat_entropy)
    return df


def add_geometry_score(df):
    """geometry_inv's scored value: -predicted_clearance_m, higher = riskier, same sign
    convention as introspection.py::SIGNALS["geometry_inv"]."""
    df = df.copy()
    df["geometry_score"] = -1.0 * df["predicted_clearance_m"]
    return df


def add_label_k1(df):
    df = df.copy()
    df["label_k1"] = (df["engine_collision_flag"].fillna(0) > 0).astype(int)
    return df


def roc_auc(y, scores):
    """Rank-based AUC, identical formula to introspection.py::roc_auc; duplicated for
    the same reason as the loaders above."""
    sub = pd.DataFrame({"y": y, "s": scores}).dropna()
    if len(sub) < 3 or sub["y"].nunique() < 2:
        return None, int(len(sub))
    yv = sub["y"].to_numpy()
    n_pos = int(yv.sum())
    n_neg = len(yv) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None, int(len(sub))
    ranks = sub["s"].rank(method="average").to_numpy()
    sum_ranks_pos = float(ranks[yv == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), int(len(sub))


def _standardize(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def fit_logistic(X, y, n_iter=25, ridge=1e-6):
    """Newton-Raphson / IRLS for a low-dimensional logistic regression (intercept + 2
    features here) -- exact to float precision for a well-conditioned design matrix,
    a handful of iterations to converge. `ridge` is a small L2 term for numerical safety
    only (near-perfect separation would otherwise blow up the Newton step)."""
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    d = Xb.shape[1]
    beta = np.zeros(d)
    for _ in range(n_iter):
        z = Xb @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        w = np.clip(p * (1 - p), 1e-6, None)
        grad = Xb.T @ (y - p) - ridge * beta
        hess = -(Xb.T * w) @ Xb - ridge * np.eye(d)
        step = np.linalg.solve(hess, grad)
        beta = beta - step
        if np.max(np.abs(step)) < 1e-8:
            break
    z = Xb @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
    return beta, p


def analyze_arm(agent, benchmark):
    df = _load_steps(agent, benchmark)
    if df is None:
        return None
    df = add_entropy_unified(df)
    df = add_geometry_score(df)
    df = add_label_k1(df)
    dec = df[df["is_decision"] == 1].reset_index(drop=True)
    sub = dec[["entropy_unified", "geometry_score", "label_k1"]].dropna()

    out = {
        "regime": REGIME,
        "n_decision_steps": int(len(dec)),
        "n_usable_rows": int(len(sub)),
        "combination": None,
    }
    if len(sub) < 50 or sub["label_k1"].nunique() < 2:
        return out

    y = sub["label_k1"].to_numpy().astype(float)
    x_int = sub["entropy_unified"].to_numpy()
    x_geo = sub["geometry_score"].to_numpy()
    X = np.column_stack([_standardize(x_int), _standardize(x_geo)])
    beta, p_combined = fit_logistic(X, y)

    auc_internal, _ = roc_auc(y, x_int)
    auc_geometry, _ = roc_auc(y, x_geo)
    auc_combined, _ = roc_auc(y, p_combined)

    out["combination"] = {
        "auroc_internal_entropy_alone": auc_internal,
        "auroc_geometry_alone": auc_geometry,
        "auroc_combined_in_sample_upper_bound": auc_combined,
        "combined_minus_geometry_alone": (
            float(auc_combined - auc_geometry)
            if auc_combined is not None and auc_geometry is not None else None
        ),
        "logistic_coefficients_standardized": {
            "intercept": float(beta[0]),
            "entropy_unified_z": float(beta[1]),
            "geometry_score_z": float(beta[2]),
        },
    }
    return out


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
    ap.add_argument("--out", default=os.path.join(REPO, "results", "summary", "signal_completeness.json"))
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
