#!/usr/bin/env python3
"""results/summary/*.json -> outputs/tables/generated/{tab_*.tex, numbers_macros.tex}
.

What: one function per table in the paper's measure -> predict -> prevent order, plus the
`\\pn{key}` macros (analysis/prose_numbers.py) every number in the prose goes through. A cell is a number when its
source JSON has the arm; `\\nmark` when the agent's config declares the cell in
`unavailable:` (or, for the guard column, the agent is outside the method set); otherwise
`\\TBD` -- a run not yet done, never a number looked up by hand. `paper/tables/tab_*.tex`
are thin `\\input{tables/generated/...}` wrappers, so the LaTeX tree never carries a typed
number.

  tab_coverage  agents, interface tier, benchmarks, guard arm       paper_numbers.json
  tab_profile   P1 + P2 per agent, both regimes                     paper_numbers.json
  tab_signals   AUROC per signal source, stall recall, swap room    introspection.json, headroom.json
  tab_guard     the method on every agent, paired off vs on         paper_numbers.json (guard)
  tab_ablation  ETP-R1: guard components, slide regime              paper_numbers.json (guard)

Agents in PLANNED have no config and no arm yet; their rows render as `\\TBD`. The
released set is empty: every row in the paper has its arms under results/.

Usage: analysis/make_tables.py
Modern Python (3.9+); never imported by the harness.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SUMMARY = os.path.join(REPO, "results", "summary")
OUTDIR = os.path.join(REPO, "outputs", "tables", "generated")

sys.path.insert(0, REPO)
from vlnsafe import config as cfg  # noqa: E402

TBD = r"\TBD"
NMARK = r"\nmark"
METHOD = "stall-mem1"      # configs/guard/stall-mem1.yaml, the method config (Decisions table)

# Static bibliographic metadata: not a run output, so not in paper_numbers.json, but not
# typed into a table either -- widening the model set is a one-line edit here.
AGENT_META = {
    "etpr1":     {"cite": "ye2025etpr1", "family": "topological, RL", "year": 2025},
    "etpnav":    {"cite": "an2025etpnav", "family": "topological", "year": 2024},
    "streamvln": {"cite": "wei2025streamvln", "family": "navigation VLA", "year": 2025},
    # waypoint-head VLA; its waypoints are walked out as primitives
    "omninav":   {"cite": "xue2025omninav", "family": "navigation VLA", "year": 2026},
    "cma":       {"cite": "krantz2020vlnce", "family": "recurrent", "year": 2020},
    "seq2seq":   {"cite": "krantz2020vlnce", "family": "recurrent", "year": 2020},
}
PLANNED = {}
# Current architecture first within each tier, T2 before T1.
ORDER = ["etpr1", "etpnav", "omninav", "streamvln", "cma", "seq2seq"]
# Agents the guard is run on. Seq2Seq is outside the method set.
GUARD_SET = ["etpr1", "etpnav", "omninav", "streamvln", "cma"]
RXR_ORDER = ["etpr1", "etpnav"]      # the RxR generality check (Decisions: R2R headline)
# "No new RxR arms" (Decisions table, 2026-09-21): these RxR cells are out of scope, not
# pending, so they render as \nmark rather than a red cell that would never fill.
RXR_OUT_OF_SCOPE = {"streamvln", "cma", "omninav"}


def _load(name):
    path = os.path.join(SUMMARY, name)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_agents():
    agents = {a: dict(cfg.load_agent(a), **AGENT_META[a]) for a in AGENT_META}
    agents.update(PLANNED)
    return agents


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def unavailable(agent_cfg, key):
    return (agent_cfg.get("unavailable") or {}).get(key)


# ------------------------------------------------------------------------- formatting

def _minus(s):
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


def pct(x, nd=1):
    return TBD if x is None else "{:.{nd}f}".format(100.0 * x, nd=nd)


def spct(x, nd=1):
    """Signed percentage points from a fraction delta."""
    return TBD if x is None else _minus("{:+.{nd}f}".format(100.0 * x, nd=nd))


def num(x, nd=1):
    return TBD if x is None else _minus("{:.{nd}f}".format(x, nd=nd))


def snum(x, nd=1):
    return TBD if x is None else _minus("{:+.{nd}f}".format(x, nd=nd))


def frac3(x):
    if x is None:
        return TBD
    s = "{:.3f}".format(x)
    return s[1:] if s.startswith("0.") else s


def star(metric):
    """`*` when the paired 95% CI excludes zero."""
    lo, hi = (metric or {}).get("ci95") or (None, None)
    return r"$^{\ast}$" if lo is not None and (lo > 0 or hi < 0) else ""


def tier_cmd(a):
    return r"\Ttwo{}" if a["tier"] == "T2" else r"\Tone{}"


def write(name, body):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "{}.tex".format(name)), "w") as f:
        f.write("% Machine-written by analysis/make_tables.py -- do not edit by hand.\n")
        f.write(body)


def table(label, caption, colspec, header, body, star_env=False, sep="3pt"):
    env = "table*" if star_env else "table"
    return (r"""\begin{%s}[t]
\centering\small
\setlength{\tabcolsep}{%s}
\caption{%s}
\label{%s}
\begin{tabular}{@{}%s@{}}
\toprule
%s
\midrule
%s
\bottomrule
\end{tabular}
\end{%s}
""" % (env, sep, caption, label, colspec, header, body, env))


# --------------------------------------------------------------------------- coverage

def published_sr(agent_cfg):
    """R2R-CE val-unseen SR the agent's own paper reports: the config's `published:` block
    (the reproduction-gate target), else the survey sheet by citation key (planned agents)."""
    sr = get(agent_cfg, "published", "r2r", "sr")
    if sr is not None or "published" in agent_cfg:
        return sr
    import csv
    with open(os.path.join(REPO, "survey", "reported_sr_r2rce.csv")) as f:
        for row in csv.DictReader(f):
            if row["key"] == agent_cfg["cite"] and row["sr_reported"]:
                return float(row["sr_reported"])
    return None


def make_coverage(numbers, agents):
    rows, n_models, n_t2 = [], 0, 0
    for aid in ORDER:
        a = agents[aid]
        pub = published_sr(a)
        slide = get(numbers, aid, "r2r", "slide")
        cells = [NMARK if pub is None else "{:.1f}".format(pub),
                 pct(slide["n_success"] / slide["n_episodes"]) if slide else TBD]
        if "rxr" in numbers.get(aid, {}):
            cells.append(r"\cmark")
        else:
            cells.append(NMARK if unavailable(a, "rxr_arm") or aid in RXR_OUT_OF_SCOPE else TBD)
        if aid not in GUARD_SET:
            cells.append(NMARK)
        else:
            cells.append(r"\cmark" if get(numbers, aid, "r2r", "guard", "block", METHOD) else TBD)
        if slide or "rxr" in numbers.get(aid, {}):
            n_models += 1
            n_t2 += a["tier"] == "T2"
        rows.append(r"{}~\cite{{{}}} & {} & {} & {} & {} \\".format(
            a["display_name"], a["cite"], a["family"], a["year"], tier_cmd(a),
            " & ".join(cells)))
        if aid == "etpnav":
            rows.append(r"\midrule")
    write("tab_coverage", table(
        "tab:coverage",
        r"""Agents, all run frozen from their authors' released checkpoints. Tier is
the decision interface (Section~\ref{sec:interfaces}): \Tone{} scores primitive
actions, \Ttwo{} scores spatial subgoals. SR: R2R-CE validation-unseen success rate
(\%) as published and as reproduced by our harness under \slideon{}, the regime the
published numbers use. Guard: the method on R2R-CE (Section~\ref{sec:guard}).
\nmark{}: no published number or public RxR-CE checkpoint, or outside the method set;
red cells are runs not yet done.""",
        "lllccccc",
        r"& & & & \multicolumn{2}{c}{SR} & & \\" "\n" r"\cmidrule(lr){5-6}" "\n"
        r"Agent & Family & Year & Tier & pub. & ours & RxR & Guard \\",
        "\n".join(rows), sep="2.5pt"))
    write("coverage_macros", "\\newcommand{{\\NumModels}}{{{}}}\n\\newcommand{{\\NumTtwo}}{{{}}}\n"
          .format(n_models, n_t2))


# ---------------------------------------------------------------------------- profile

def profile_cells(numbers, agents, aid, bench):
    blk, sld = (get(numbers, aid, bench, r) for r in ("block", "slide"))

    def pair(f):
        return [f(blk) if blk else TBD, f(sld) if sld else TBD]

    sr = pair(lambda x: pct(x["n_success"] / x["n_episodes"]))
    dsr = pair(lambda x: pct(x["p1"]["dirty_success_rate"]))
    nc = pair(lambda x: pct(x["p1"]["nc_rate_mean"]))
    stuck = pair(lambda x: pct(x["p1"]["stuck_rate"]))
    d = get(numbers, aid, bench, "regime_delta")
    dsr_d = TBD if d is None else _minus("{:+.1f}".format(d["delta_sr_pp"]))
    miss = TBD if d is None or d["kappa_miss"] is None else pct(d["kappa_miss"])
    return sr + [dsr_d] + nc + dsr + stuck + [miss]


def make_profile(numbers, agents):
    def block(bench, order, label):
        lines = [r"\multicolumn{{12}}{{@{{}}l}}{{\textit{{{}}}}}\\".format(label)]
        for aid in order:
            a = agents[aid]
            lines.append("{} & {} & {} \\\\".format(
                a["display_name"], tier_cmd(a), " & ".join(profile_cells(numbers, agents, aid, bench))))
        return "\n".join(lines)

    header = (r"& & \multicolumn{3}{c}{Success rate} & \multicolumn{2}{c}{\NC{} rate}"
              r" & \multicolumn{2}{c}{Dirty success} & \multicolumn{2}{c}{Stuck}"
              r" & $\hat{\kappa}$ miss \\" "\n"
              r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}"
              r"\cmidrule(l){12-12}" "\n"
              r"Agent & Tier & \slideoff{} & \slideon{} & $\Delta$ & \slideoff{} & \slideon{}"
              r" & \slideoff{} & \slideon{} & \slideoff{} & \slideon{} & \slideon{} \\")
    body = block("r2r", ORDER, "R2R-CE, 1{,}839 episodes") + "\n\\midrule\n" + \
        block("rxr", RXR_ORDER, "RxR-CE, 1{,}223 episodes")
    write("tab_profile", table(
        "tab:profile",
        r"""Contact by outcome (P1) and its sensitivity to evaluation choices (P2), all
in \%. Dirty success is $\Pr[\text{contact}\mid\text{success}]$ under the engine flag
$\kappa_\rho$; stuck is the share of episodes with at least one proxy stuck event;
$\Delta$ is the success-rate gain from \slideoff{} to \slideon{} on the same episodes;
$\hat{\kappa}$ miss is the share of engine-contact episodes the displacement proxy
classifies clean, under \slideon{}. Red cells are runs not yet done.""",
        "ll" + "c" * 10, header, body, star_env=True, sep="4pt"))


# ---------------------------------------------------------------------------- signals

def signal_cells(intro, headroom, aid, bench, unavailable=()):
    arm = get(intro, aid, bench)
    if arm is None:
        return [TBD] * 7
    # no scored alternatives (e.g. OmniNav's single waypoint): entropy, margin and swap
    # are undefined, drawn as \nmark with the reason in the agent config's `unavailable:`
    no_scores = "action_probabilities" in unavailable

    def auc(name):
        return get(arm, "signals_at_decision_steps", name, "3", "auroc")

    ent = auc("entropy_norm") or auc("entropy")
    vals = [auc("geometry_inv"), auc("stall_history"), ent, auc("stop_prob_inv"),
            auc("margin_inv")]
    best = max(v for v in vals if v is not None)
    cells = [(r"\textbf{%s}" % frac3(v)) if v == best else frac3(v) for v in vals]
    cells.append(pct(get(arm, "stall_history_forward", "recall")))
    cells.append(pct(get(headroom, aid, bench, "eps_sweep", "0.1", "headroom_rate_unconditional")))
    if no_scores:
        cells[2] = cells[4] = cells[6] = NMARK
    return cells


def make_signals(intro, headroom, agents):
    def block(bench, order, label):
        lines = [r"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{}}}}}\\".format(label)]
        for aid in order:
            a = agents[aid]
            mark = r"$^{\ast}$" if a["tier"] == "T1" else ""
            lines.append("{}{} & {} \\\\".format(
                a["display_name"], mark, " & ".join(signal_cells(intro, headroom, aid, bench,
                                       a.get("unavailable") or {}))))
        return "\n".join(lines)

    header = (r"& \multicolumn{2}{c}{External} & \multicolumn{3}{c}{Policy} & & \\" "\n"
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}" "\n"
              r"Agent & Geom. & Hist. & Ent. & Stop & Marg. & Hist.\,rec. & Swap \\")
    body = block("r2r", ORDER, "R2R-CE") + "\n\\midrule\n" + block("rxr", RXR_ORDER, "RxR-CE")
    write("tab_signals", table(
        "tab:signals",
        r"""Which information predicts contact (\slideoff{}). AUROC for an engine
collision within three primitives of a decision, from depth geometry, movement history
(the agent stands where a forward already failed), and the policy's own candidate
entropy, stop probability and top-2 margin; the best signal per row is bold.
Hist.\,rec.: share of collided forward steps that movement history flags (\%); every
flagged step collided. Swap: share of risky decisions with a geometrically safe
alternative within 10 score points (\%); $^{\ast}$\Tone{} alternatives are primitives.
Red cells are runs not yet done.""",
        "lccccccc", header, body, sep="2.5pt"))


# ------------------------------------------------------------------------------ guard

def guard_cells(numbers, aid, name=METHOD, regime="block"):
    g = get(numbers, aid, "r2r", "guard", regime, name)
    if g is None:
        return [TBD] * 13
    m = g["metrics"]

    def om(key, f):
        return [f(m[key]["off"]), f(m[key]["on"])]

    return (om("success", pct) + [spct(m["success"]["delta"]) + star(m["success"])]
            + [spct(m["spl"]["delta"]) + star(m["spl"]), spct(m["ndtw"]["delta"]) + star(m["ndtw"])]
            + om("collision_count", num) + om("collision_event_count", lambda x: num(x, 2))
            + om("stuck_flag", pct)
            + ["{}/{}".format(g["success_flips"]["gained"], g["success_flips"]["lost"]),
               pct(g["intervention"]["guard_owned_step_share"])])


def make_guard(numbers, agents):
    rows = []
    for aid in GUARD_SET:
        a = agents[aid]
        rows.append("{} & {} & {} \\\\".format(a["display_name"], tier_cmd(a),
                                                " & ".join(guard_cells(numbers, aid))))
        if aid == "etpnav":
            rows.append(r"\midrule")
    header = (r"& & \multicolumn{3}{c}{Success rate} & SPL & nDTW"
              r" & \multicolumn{2}{c}{Coll.\ steps/ep.} & \multicolumn{2}{c}{Coll.\ events/ep.}"
              r" & \multicolumn{2}{c}{Stuck (\%)} & Flips & Guard \\" "\n"
              r"\cmidrule(lr){3-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}\cmidrule(lr){8-9}"
              r"\cmidrule(lr){10-11}\cmidrule(lr){12-13}" "\n"
              r"Agent & Tier & off & on & $\Delta$ & $\Delta$ & $\Delta$ & off & on & off & on"
              r" & off & on & $+$/$-$ & steps \\")
    write("tab_guard", table(
        "tab:guard",
        r"""The collision guard on R2R-CE (\slideoff{}, 1{,}839 episodes, paired on
episode). Success rate, SPL and nDTW in \%; $\Delta$ is on minus off in points;
$^{\ast}$: the 95\% bootstrap interval over episodes excludes zero. Flips: episodes the
guard turns from failure to success ($+$) and from success to failure ($-$). Guard
steps: share of executed primitives the guard chose (\%). Red cells are runs not yet
done.""",
        "ll" + "c" * 13, header, "\n".join(rows), star_env=True, sep="3.5pt"))


# --------------------------------------------------------------------------- ablation

ABLATION = [  # (label, source) -- source: ("off"|"guard"|"baseline", regime, name)
    (r"No guard", ("off", "block", METHOD)),
    (r"\textbf{Guard (ours)}", ("guard", "block", METHOD)),
    (r"\quad 8-frame depth memory", ("guard", "block", "stall")),
    (r"\quad no stall veto", ("guard", "block", "default")),
    (r"\quad no stall veto, no cap $K$", ("guard", "block", "uncapped")),
]
ABLATION_SLIDE = [
    (r"No guard", ("slide", "slide", None)),
    (r"\textbf{Guard (ours)}", ("guard", "slide", METHOD)),
]
ABL_COLS = ("success", "spl", "collision_count", "collision_event_count", "stuck_flag")


def ablation_cells(numbers, source):
    kind, regime, name = source
    if kind == "guard" or kind == "off":
        g = get(numbers, "etpr1", "r2r", "guard", regime, name)
        if g is None:
            return [TBD] * 5
        vals = {k: g["metrics"][k]["on" if kind == "guard" else "off"] for k in ABL_COLS}
    else:
        arm = (get(numbers, "etpr1", "r2r", "baseline", name) if kind == "baseline"
               else get(numbers, "etpr1", "r2r", regime))
        if arm is None:
            return [TBD] * 5
        vals = arm["episode_means"]
    return [pct(vals["success"]), pct(vals["spl"]), num(vals["collision_count"]),
            num(vals["collision_event_count"], 2), pct(vals["stuck_flag"])]


def make_ablation(numbers):
    def block(spec, label):
        lines = [r"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{}}}}}\\".format(label)]
        lines += ["{} & {} \\\\".format(lab, " & ".join(ablation_cells(numbers, src)))
                  for lab, src in spec]
        return "\n".join(lines)

    body = block(ABLATION, r"\slideoff{}") + "\n\\midrule\n" + block(ABLATION_SLIDE, r"\slideon{}")
    write("tab_ablation", table(
        "tab:ablation",
        r"""ETP-R1 on R2R-CE: the guard's components, and the guard under \slideon{}. SR, SPL and stuck in \%; collision
steps and events per episode. Indented rows remove one component from the row above.
Red cells are runs not yet done.""",
        "lccccc", r"Condition & SR & SPL & Steps & Events & Stuck \\", body, sep="3pt"))


def main():
    numbers = _load("paper_numbers.json")
    intro = _load("introspection.json")
    headroom = _load("headroom.json")
    agents = load_agents()
    make_coverage(numbers, agents)
    make_profile(numbers, agents)
    make_signals(intro, headroom, agents)
    make_guard(numbers, agents)
    make_ablation(numbers)
    print("wrote {}/tab_{{coverage,profile,signals,guard,ablation}}.tex, coverage_macros.tex"
          .format(OUTDIR))


if __name__ == "__main__":
    main()
