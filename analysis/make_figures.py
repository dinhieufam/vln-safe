#!/usr/bin/env python3
"""results/summary/paper_numbers.json + results/summary/headroom.json ->
outputs/figures/*.pdf.

What: the three figures the paper actually includes (fig_autopsy, fig_metric,
fig_headroom -- results.tex's only \\includegraphics calls). Each is the \\Ttwo{}
headline pair, current architecture (ETP-R1) beside its predecessor (ETPNav), the same
pairing tab_backbone/tab_regime/tab_auroc/tab_headroom use. Both are fully landed as of
this pass, so unlike the pipeline this was ported from there is no
placeholder/TBD panel left to draw -- if a future rerun finds one side missing, this
script stops with a clear error rather than silently shipping a blank or stale PDF.

Usage: analysis/make_figures.py
Modern Python (3.9+, matplotlib, numpy); never imported by the harness.

Ported from vln_research/pipeline/figures/make_figures.py (348 l, single-agent
`paper_stats.json`/`audit_stats.json`, ETP-R1 panels drawn as hatched TBD placeholders).
fig_audit and fig_auroc were already dropped from the paper there (Table 1 and Table 7
carry that content instead) and stay dropped here. Condition keys/labels move from the
old `r2r_no_sliding` naming to the current (benchmark, regime) pairs and the paper's own
\\slideoff{}/\\slideon{} vocabulary.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIGDIR = os.path.join(REPO, "outputs", "figures")
SUMMARY = os.path.join(REPO, "results", "summary")

# Categorical slots 1-4 of the reference palette (validated light-mode, dataviz skill).
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e3e2dd"

CURRENT_ID, CURRENT_LABEL = "etpr1", "ETP-R1 (2025, current)"
PREDECESSOR_ID, PREDECESSOR_LABEL = "etpnav", "ETPNav (2024, predecessor)"

COL, FULL = 3.33, 7.0          # AAMAS sigconf column and text width, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.edgecolor": INK3,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
})

CONDS = [("r2r", "block"), ("r2r", "slide"), ("rxr", "block"), ("rxr", "slide")]
# matplotlib is not rendering with usetex here, so labels spell out the paper's
# \slideoff{}/\slideon{} vocabulary (block/slide) in plain text rather than the macros.
CLABELS_PLAIN = ["R2R-CE\nblock", "R2R-CE\nslide", "RxR-CE\nblock", "RxR-CE\nslide"]


def load(name):
    with open(os.path.join(SUMMARY, name)) as f:
        return json.load(f)


def arm(numbers, agent, benchmark, regime):
    a = numbers.get(agent, {}).get(benchmark, {}).get(regime)
    if a is None:
        raise SystemExit(
            "make_figures.py needs {}/{}/{}, which is not in paper_numbers.json -- "
            "run analysis/collect.py first, or if the arm genuinely has not landed yet, "
            "this script has no placeholder for it any more (see module docstring)".format(
                agent, benchmark, regime))
    return a


def bare(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------------- fig_autopsy
def fig_autopsy(numbers):
    """The joint outcome-by-contact distribution, one panel per \\Ttwo{} agent."""
    segs = [
        ("clean_success", "Clean success", S1),
        ("dirty_success", "Dirty success", S2),
        ("clean_failure", "Clean failure", S3),
        ("dirty_failure", "Dirty failure", S4),
    ]
    x = np.arange(len(CONDS))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.95), sharey=True,
                             gridspec_kw={"wspace": 0.08})

    for ax, agent, title in [(axes[0], CURRENT_ID, CURRENT_LABEL),
                              (axes[1], PREDECESSOR_ID, PREDECESSOR_LABEL)]:
        bottom = np.zeros(len(CONDS))
        arms = [arm(numbers, agent, b, r) for b, r in CONDS]
        for key, name, color in segs:
            vals = np.array([a["joint_rates"][key] for a in arms])
            ax.bar(x, vals, 0.6, bottom=bottom, color=color, label=name,
                   edgecolor="white", linewidth=1.4, zorder=3)
            for xi, v, b in zip(x, vals, bottom):
                if v > 0.045:  # relief rule: label every readable band
                    ax.text(xi, b + v / 2, f"{v*100:.0f}%", ha="center", va="center",
                            fontsize=6.5, color="white", zorder=4)
            bottom += vals
        # share of successes that are dirty -- the headline, called out above each bar
        # because it is a conditional rate, not a stack segment.
        for xi, a in zip(x, arms):
            d = a["p1"]["dirty_success_rate"]
            lo, hi = a["p1"]["dirty_success_ci95"]
            ax.text(xi, 1.105, f"{d*100:.1f}% dirty", ha="center", va="bottom",
                    fontsize=6.5, color=INK)
            ax.text(xi, 1.035, f"[{lo*100:.1f}, {hi*100:.1f}]", ha="center", va="bottom",
                    fontsize=5.8, color=INK3)
        ax.set_title(title, fontsize=7, color=INK, pad=14)
        ax.set_xticks(x, CLABELS_PLAIN, fontsize=6.5)
        ax.set_ylim(0, 1.22)
        bare(ax)
        ax.spines["left"].set_bounds(0, 1)

    axes[0].set_yticks(np.arange(0, 1.01, 0.25), ["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("share of episodes", fontsize=6.5)
    axes[1].tick_params(axis="y", length=0)
    hh, ll = axes[1].get_legend_handles_labels()
    fig.legend(hh, ll, frameon=False, fontsize=6.5, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.20), handlelength=1.1, labelcolor=INK2,
               columnspacing=1.4)
    fig.savefig(os.path.join(FIGDIR, "fig_autopsy.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- fig_metric
def fig_metric(numbers):
    """Two collision definitions, one set of episodes: the comparability problem."""
    series = [("dirty_success_rate", "Physics-engine contact flag", S1, "p1"),
              ("dirty_success_rate_proxy", "Displacement proxy", S2, "p2")]
    x = np.arange(len(CONDS))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(COL * 2, 1.95), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax in axes:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)

    for ax, agent, title in [(axes[0], CURRENT_ID, "ETP-R1 (current)"),
                              (axes[1], PREDECESSOR_ID, "ETPNav (predecessor)")]:
        arms = [arm(numbers, agent, b, r) for b, r in CONDS]
        for i, (key, name, color, block) in enumerate(series):
            vals = [a[block][key] for a in arms]
            off = (i - 0.5) * (w + 0.02)
            ax.bar(x + off, vals, w, label=name, color=color, edgecolor="white",
                   linewidth=1.0, zorder=3)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 0.018, f"{v*100:.0f}", ha="center", va="bottom",
                        fontsize=6, color=INK2, zorder=4)
        ax.set_title(title, fontsize=7, color=INK, pad=3)
        ax.set_xticks(x, ["R2R\nblock", "R2R\nslide", "RxR\nblock", "RxR\nslide"], fontsize=6.2)
        ax.set_ylim(0, 0.96)
        bare(ax)

    axes[0].set_yticks(np.arange(0, 0.81, 0.2), [f"{int(v*100)}%" for v in np.arange(0, 0.81, 0.2)])
    axes[0].set_ylabel("dirty-success rate", fontsize=6.5)
    axes[1].tick_params(axis="y", length=0)
    hh, ll = axes[1].get_legend_handles_labels()
    fig.legend(hh, ll, frameon=False, fontsize=6.3, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.13), handlelength=1.1, labelcolor=INK2,
               columnspacing=1.2)
    fig.savefig(os.path.join(FIGDIR, "fig_metric.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------- fig_headroom
def fig_headroom(headroom):
    """Usable headroom against the score-margin budget, block regime (the regime P3/P4
    are computed under). ETPNav (measured, solid) and ETP-R1
    (measured too as of this pass, dashed overlay) on the same axes."""
    lines = [
        ("availability_rate_among_risky", "An alternative exists", S1, "o"),
        ("headroom_rate_conditional_on_availability", "...and it is geometrically safe", S2, "s"),
        ("headroom_rate_unconditional", "Usable headroom", S3, "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.95), sharey=True)
    for ax, benchmark, title in zip(axes, ["r2r", "rxr"], ["R2R-CE val-unseen", "RxR-CE val-unseen"]):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)

        predecessor = headroom[PREDECESSOR_ID][benchmark]["eps_sweep"]
        current = headroom[CURRENT_ID][benchmark]["eps_sweep"]
        eps = sorted(float(k) for k in predecessor)

        for key, name, color, mk in lines:
            vals = [predecessor[f"{e:g}"][key] * 100 for e in eps]
            ax.plot([e * 100 for e in eps], vals, color=color, linewidth=2.0,
                    marker=mk, markersize=3.6, markeredgecolor="white",
                    markeredgewidth=0.7, label=name, zorder=3)
            cvals = [current[f"{e:g}"][key] * 100 for e in eps]
            ax.plot([e * 100 for e in eps], cvals, color=color, linewidth=1.4,
                    linestyle="--", marker=mk, markersize=2.6, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=0.7, zorder=3, alpha=0.85)

        ax.axvline(10, color=GRID, linewidth=1.0, zorder=2)
        h = predecessor["0.1"]["headroom_rate_unconditional"] * 100
        h_cur = current["0.1"]["headroom_rate_unconditional"] * 100
        ax.text(10.8, 57, f"at $\\varepsilon{{=}}10$pp, usable headroom is\n"
                          f"{h:.1f}% (ETPNav), {h_cur:.1f}% (ETP-R1)",
                fontsize=6.3, color=INK2, va="top", ha="left")
        ax.set_title(f"{title}\nsolid: ETPNav, dashed: ETP-R1", fontsize=7, color=INK, pad=3)
        ax.set_xlabel("score-margin budget $\\varepsilon$ the layer may spend (pp)", fontsize=6.5)
        bare(ax)

    axes[0].set_ylabel("percent of risky decisions", fontsize=6.5)
    axes[0].set_ylim(0, 60)
    axes[0].set_yticks(np.arange(0, 61, 15), [f"{v}%" for v in np.arange(0, 61, 15)])
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, frameon=False, fontsize=6.5, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.10), handlelength=1.4, labelcolor=INK2,
               columnspacing=1.6)
    fig.savefig(os.path.join(FIGDIR, "fig_headroom.pdf"))
    plt.close(fig)


#  ------------------------------------------------------------------- fig_contact_timing
# R2R-CE agents in the order of the paper's tables (topological, then VLA, then recurrent).
TIMING_AGENTS = [("etpr1", "ETP-R1"), ("etpnav", "ETPNav"), ("omninav", "OmniNav"),
                 ("streamvln", "StreamVLN"), ("cma", "CMA"), ("seq2seq", "Seq2Seq")]


FIRST_CONTACT = os.path.join(SUMMARY, "first_contact_fractions.json")


def first_contact_fraction(agent, regime="slide"):
    """Position of the first engine collision in each dirty success, as a fraction of the
    episode's executed primitives (0 = first primitive, 1 = the STOP).

    Needs the arm's steps.csv.gz; the fractions are cached in FIRST_CONTACT so the figure
    can be redrawn from the released results without the step logs."""
    import pandas as pd
    key = "{}/r2r/{}".format(agent, regime)
    d = os.path.join(REPO, "results", agent, "r2r", regime)
    steps_path = os.path.join(d, "steps.csv.gz")
    cache = load("first_contact_fractions.json") if os.path.exists(FIRST_CONTACT) else {}
    if not os.path.exists(steps_path):
        return np.array(cache[key])
    eps = pd.read_csv(os.path.join(d, "episodes.csv"))
    steps = pd.read_csv(steps_path,
                        usecols=["episode_id", "step_index", "engine_collision_flag"])
    steps = steps.drop_duplicates(["episode_id", "step_index"])
    first = steps[steps.engine_collision_flag == 1].groupby("episode_id").step_index.min()
    dirty = eps[(eps.success == 1) & (eps.collision_count > 0)].set_index("episode_id")
    fracs = (first.reindex(dirty.index) / dirty.step_count).dropna().to_numpy()
    cache[key] = [round(float(f), 6) for f in fracs]
    with open(FIRST_CONTACT, "w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    return fracs


def fig_contact_timing(numbers):
    """(a) outcome x contact under slide for every R2R-CE agent; (b) where in a dirty
    success the first contact happens. Agents share the y axis, so identity is the axis
    label, never a color."""
    segs = [("clean_success", "Clean success", S1), ("dirty_success", "Dirty success", S2),
            ("clean_failure", "Clean failure", S3), ("dirty_failure", "Dirty failure", S4)]
    y = np.arange(len(TIMING_AGENTS))[::-1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COL, 1.9), sharey=True,
                                 gridspec_kw={"wspace": 0.32, "width_ratios": [1.2, 1]})
    left = np.zeros(len(TIMING_AGENTS))
    arms = [arm(numbers, a, "r2r", "slide") for a, _ in TIMING_AGENTS]
    for key, name, color in segs:
        vals = np.array([a["joint_rates"][key] for a in arms])
        ax.barh(y, vals, 0.62, left=left, color=color, label=name, edgecolor="white",
                linewidth=1.0, zorder=3)
        left += vals
    for yi, a in zip(y, arms):
        ax.text(1.02, yi, f"{a['p1']['dirty_success_rate']*100:.0f}%", va="center",
                ha="left", fontsize=6, color=INK)
    ax.text(1.02, y[0] + 0.45, "DSR", va="bottom", ha="left", fontsize=6, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "50", "100%"])
    ax.set_xlabel("(a) episodes under slide", fontsize=6.5)
    bare(ax)

    fracs = [first_contact_fraction(a) for a, _ in TIMING_AGENTS]
    bp = bx.boxplot(fracs, positions=y, orientation="horizontal", widths=0.5, whis=(10, 90),
                    showfliers=False, patch_artist=True, zorder=3)
    for box in bp["boxes"]:
        box.set(facecolor=S2, alpha=0.35, edgecolor=S2, linewidth=0.8)
    for part in ("whiskers", "caps"):
        for line in bp[part]:
            line.set(color=INK3, linewidth=0.7)
    for line in bp["medians"]:
        line.set(color=INK, linewidth=1.2)
    bx.set_xlim(0, 1)
    bx.set_xticks([0, 0.5, 1], ["start", "mid", "stop"])
    bx.set_xlabel("(b) first-contact position", fontsize=6.5)
    ax.set_yticks(y, [lab for _, lab in TIMING_AGENTS], fontsize=6.5)
    bx.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    bx.tick_params(axis="y", length=0)
    bare(bx, ("top", "right", "left"))
    hh, ll = ax.get_legend_handles_labels()
    fig.legend(hh, ll, frameon=False, fontsize=6, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.93), handlelength=0.9, columnspacing=0.8,
               labelcolor=INK2)
    fig.savefig(os.path.join(FIGDIR, "fig_contact_timing.pdf"))
    plt.close(fig)
    return {a: float(np.median(f)) for (a, _), f in zip(TIMING_AGENTS, fracs)}


FIGURES = ("autopsy", "metric", "headroom", "contact_timing")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=FIGURES, action="append",
                    help="draw only these figures (repeatable); default: all")
    which = set(ap.parse_args().only or FIGURES)
    numbers = load("paper_numbers.json")
    os.makedirs(FIGDIR, exist_ok=True)
    if "autopsy" in which:
        fig_autopsy(numbers)
    if "metric" in which:
        fig_metric(numbers)
    if "headroom" in which:
        fig_headroom(load("headroom.json"))
    if "contact_timing" in which:
        print("median first-contact position:", fig_contact_timing(numbers))
    print("wrote:", *(p for p in sorted(os.listdir(FIGDIR)) if p.startswith("fig_")))


if __name__ == "__main__":
    main()
