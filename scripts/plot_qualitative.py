#!/usr/bin/env python3
"""data/scratch/qualitative/*.npz (scripts/render_qualitative.py) ->
outputs/figures/supp/fig_qualitative_maps.pdf and fig_qualitative_frames.pdf.

Maps: one panel per case, the unguarded and the guarded path of the same episode over the
navigation mesh, with collision steps, guard vetoes and escape segments marked.
Frames: the egocentric view from which the policy proposed a vetoed forward step, and the
view after the guard's escape, for the VLA rescue case.

Modern Python (matplotlib, numpy); never imported by the harness.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "data", "scratch", "qualitative")
OUT = os.path.join(REPO, "outputs", "figures", "supp")

OFF, ON = "#eb6834", "#2a78d6"          # reference palette slots 2 and 1
INK, INK2, FLOOR = "#0b0b0b", "#52514e", "#ecebe6"
NAMES = {"etpr1": "ETP-R1", "etpnav": "ETPNav", "omninav": "OmniNav"}
TITLES = {
    "rescue_topological": "Stuck failure rescued",
    "rescue_vla": "Stuck failure rescued",
    "clean_under_slide": "Dirty success made clean",
    "residual_escape_contact": "Residual contact on an escape",
    "guard_failure": "Success turned into failure",
}

plt.rcParams.update({"font.family": "serif", "font.size": 7, "pdf.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def outcome(c, arm):
    r = c[arm]
    return "{}, {} steps, {} coll.".format("success" if r["success"] else "failure",
                                          r["steps"], r["collisions"])


def draw_map(ax, c, d):
    b = d["bounds"]
    ax.imshow(d["topdown"] > 0, cmap=matplotlib.colors.ListedColormap(["white", FLOOR]),
              origin="upper", extent=[b[0][0], b[1][0], b[1][1], b[0][1]],
              interpolation="nearest", zorder=0)
    pts = []
    for arm, col, z in (("off", OFF, 2), ("on", ON, 3)):
        p, hits = d["path_" + arm], d["hits_" + arm]
        xz = p[:, [0, 2]]
        pts.append(xz)
        ax.plot(xz[:, 0], xz[:, 1], "-", color=col, lw=1.3, zorder=z, solid_capstyle="round")
        hi = np.where(hits)[0] + 1   # path[j] is the pose after step j-1
        ax.plot(xz[hi, 0], xz[hi, 1], "x", color=col, ms=4.5, mew=1.2, zorder=z + 2)
    g = d["guard_on"]
    xz = d["path_on"][:, [0, 2]]
    esc = np.where(g == "maneuver")[0] + 1
    ax.plot(xz[esc, 0], xz[esc, 1], "o", ms=2.2, color=ON, mfc="white", mew=0.7, zorder=5)
    veto = np.where(np.isin(g, ["abort", "stall"]))[0]
    ax.plot(xz[veto, 0], xz[veto, 1], "^", ms=4.5, color=INK, mfc="#f2c94c", mew=0.6, zorder=6)
    start, goal = pts[0][0], d["goal"]
    ax.plot(*start, "s", ms=5, color=INK, zorder=7)
    ax.plot(goal[0], goal[2], "*", ms=10, color="#f2c94c", mec=INK, mew=0.6, zorder=7)
    ax.add_patch(plt.Circle((goal[0], goal[2]), 3.0, fill=False, ls=":", lw=0.6, color=INK2))
    allp = np.vstack(pts + [goal[[0, 2]][None]])
    lo, hi = allp.min(0) - 1.0, allp.max(0) + 1.0
    span = max(hi - lo)
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax.set_ylim(mid[1] + span / 2, mid[1] - span / 2)
    ax.set_aspect("equal")
    x0, y0 = mid[0] - span / 2 + 0.05 * span, mid[1] + span / 2 - 0.05 * span
    ax.plot([x0, x0 + 1], [y0, y0], "-", color=INK, lw=2)
    ax.text(x0 + 0.5, y0 - 0.03 * span, "1 m", ha="center", va="bottom", fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c9c8c2"); s.set_linewidth(0.5)
    regime = "block" if c["regime"] == "block" else "slide"
    ax.set_title("({}) {}\n{}, {}, episode {}".format(
        c["letter"], TITLES[c["case"]], NAMES[c["agent"]], regime, c["episode_id"]),
        fontsize=7, color=INK, pad=3, linespacing=1.2)
    ax.set_xlabel("no guard: {}\nguard: {}".format(outcome(c, "off"), outcome(c, "on")),
                  fontsize=6, color=INK2, labelpad=3)


def main():
    with open(os.path.join(SRC, "cases.json")) as f:
        cases = json.load(f)
    os.makedirs(OUT, exist_ok=True)
    fig, grid = plt.subplots(2, 3, figsize=(7.0, 5.4))
    axes = grid.ravel()
    for i, (ax, c) in enumerate(zip(axes, cases)):
        c["letter"] = "abcde"[i]
        draw_map(ax, c, np.load(os.path.join(SRC, c["case"] + ".npz")))
    h = [plt.Line2D([], [], color=OFF, lw=1.3, label="no guard"),
         plt.Line2D([], [], color=ON, lw=1.3, label="guard"),
         plt.Line2D([], [], ls="", marker="x", color=INK2, label="collision step"),
         plt.Line2D([], [], ls="", marker="^", color=INK, mfc="#f2c94c", label="veto"),
         plt.Line2D([], [], ls="", marker="o", color=ON, mfc="white", ms=3, label="escape step"),
         plt.Line2D([], [], ls="", marker="s", color=INK, label="start"),
         plt.Line2D([], [], ls="", marker="*", color="#f2c94c", mec=INK, ms=8,
                    label="goal (3 m radius dotted)")]
    axes[-1].axis("off")
    axes[-1].legend(handles=h, loc="center", ncol=1, frameon=False, fontsize=7,
                    handlelength=1.6)
    fig.subplots_adjust(wspace=0.12, hspace=0.42)
    fig.savefig(os.path.join(OUT, "fig_qualitative_maps.pdf"))
    plt.close(fig)

    # frames: before the first veto / after the escape, for the VLA rescue case
    rows = [c for c in cases if c["case"] == "rescue_vla"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(3.3, 1.75 * len(rows)), squeeze=False)
    for r, c in enumerate(rows):
        d = np.load(os.path.join(SRC, c["case"] + ".npz"))
        f0, f1 = [int(x) for x in d["frames_on"]]
        for k, (step, lab) in enumerate(((f0, "policy proposes forward"),
                                         (f1, "after the guard's escape"))):
            ax = axes[r, k]
            ax.imshow(plt.imread(os.path.join(SRC, "{}_on_{}.png".format(c["case"], step))))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title("{}, step {}\n{}".format(NAMES[c["agent"]], step + 1, lab),
                         fontsize=6.5, color=INK, pad=2, linespacing=1.15)
    fig.subplots_adjust(wspace=0.04, hspace=0.18)
    fig.savefig(os.path.join(OUT, "fig_qualitative_frames.pdf"))
    plt.close(fig)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
