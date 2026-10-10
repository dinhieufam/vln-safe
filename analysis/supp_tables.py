#!/usr/bin/env python3
"""results/summary/*.json -> outputs/tables/supp/*.tex, the supplementary material's tables.

Writes only into outputs/tables/supp/ (never outputs/tables/generated/, whose tables are
hand-styled). Every cell is read from the summary JSONs; nothing
is typed.

Usage: analysis/supp_tables.py
Modern Python; never imported by the harness.
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(REPO, "results", "summary")
OUT = os.path.join(REPO, "outputs", "tables", "supp")

AGENTS = [("etpr1", "ETP-R1"), ("etpnav", "ETPNav"), ("omninav", "OmniNav"),
          ("streamvln", "StreamVLN"), ("cma", "CMA"), ("seq2seq", "Seq2Seq")]
NAME = dict(AGENTS)
GUARD = "_guard-stall-mem1"


def load(name):
    with open(os.path.join(SUMMARY, name)) as f:
        return json.load(f)


def pct(x, d=1):
    return "{:.{}f}".format(100 * x, d)


def ci(c, s=100, d=1):
    return "[{:.{d}f}, {:.{d}f}]".format(c[0] * s, c[1] * s, d=d)


def pval(p):
    if p == 0:
        return "$<10^{-300}$"
    m, e = "{:.1e}".format(p).split("e")
    return "${}\\times10^{{{}}}$".format(m, int(e))


def table(name, caption, label, colspec, header, rows, star=False, size="small", sep=3):
    env = "table*" if star else "table"
    body = "\n".join(r if r.startswith("\\midrule") or r.endswith("\\\\") else r + r" \\"
                     for r in rows)
    tex = "\n".join([
        "% Written by analysis/supp_tables.py from results/summary/*.json.",
        r"\begin{%s}[t]" % env, r"\centering\%s" % size,
        r"\setlength{\tabcolsep}{%spt}" % sep,
        r"\caption{%s}" % caption, r"\label{%s}" % label,
        r"\begin{tabular}{@{}%s@{}}" % colspec, r"\toprule", header + r" \\", r"\midrule",
        body, r"\bottomrule", r"\end{tabular}", r"\end{%s}" % env, ""])
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(tex)


def p1_tables(numbers, regime_stats):
    rows, rows2 = [], []
    for bench in ("r2r", "rxr"):
        if bench == "rxr":
            rows.append(r"\midrule")
            rows2.append(r"\midrule")
        for a, lab in AGENTS:
            arms = numbers[a].get(bench)
            if not arms or "slide" not in arms:
                continue
            lab = lab + (" (RxR)" if bench == "rxr" else "")
            for reg in ("slide", "block"):
                x = arms[reg]
                p1, jc = x["p1"], x["joint_counts"]
                rows.append(" & ".join([
                    lab if reg == "slide" else "", r"\slideon{}" if reg == "slide" else r"\slideoff{}",
                    str(jc["clean_success"]), str(jc["dirty_success"]),
                    str(jc["clean_failure"]), str(jc["dirty_failure"]),
                    pct(p1["dirty_success_rate"]) + " " + ci(p1["dirty_success_ci95"]),
                    "{:.1f}".format(p1["collisions_per_dirty_success"]),
                    "{:.0f}".format(p1["collisions_per_dirty_success_median"]),
                    pct(p1["share_of_collisions_in_successes"]),
                    pct(p1["nc_rate_success"]["mean"]), pct(p1["nc_rate_failure"]["mean"])]))
            x = arms["slide"]["p1"]
            rows2.append(" & ".join([
                lab,
                "{:.3f}".format(x["min_clearance_success"]["median"]),
                "{:.3f}".format(x["min_clearance_failure"]["median"]),
                pval(x["min_clearance_success_vs_failure"]["p"]),
                pct(x["time_in_risk_success"]["mean"]), pct(x["time_in_risk_failure"]["mean"])]))
    table("p1_full",
          r"Full P1 profile. Episode counts by outcome and contact status; DSR with its Wilson "
          r"95\% interval; collision steps per dirty success (mean, median); the share of all "
          r"collision steps that occur in successes; and the mean per-episode \NC{} rate in "
          r"successful (S) and failed (F) episodes, all in \%.",
          "tab:supp-p1", "llrrrrlrrrrr",
          r"Agent & Regime & CS & DS & CF & DF & DSR [95\% CI] & Mean & Med. & In succ. "
          r"& \NC{} S & \NC{} F", rows, star=True, size="footnotesize", sep=3)
    table("clearance",
          r"Depth-based clearance profile under \slideon{}. Median per-episode minimum "
          r"clearance (m, camera-to-surface, floor not masked) in successful (S) and failed "
          r"(F) episodes with the Mann--Whitney $U$ $p$-value, and time-in-risk (share of "
          r"primitives with clearance below 0.25\,m, \%).",
          "tab:supp-clearance", "lccccc",
          r"Agent & Min.\ clr.\ S & Min.\ clr.\ F & $p$ & Risk S & Risk F", rows2, sep=3)

    rows = []
    for key in [a + "/r2r" for a, _ in AGENTS] + ["etpr1/rxr", "etpnav/rxr"]:
        x = regime_stats[key]
        a, b = key.split("/")
        s, d, lc, sv = x["success"], x["dirty_success"], x["length_controlled"], x["survivorship"]
        rows.append(" & ".join([
            NAME[a] + (" (RxR)" if b == "rxr" else ""),
            "{:.1f} {}".format(100 * s["delta"], ci(s["ci95"])), pval(s["mcnemar_exact_p"]),
            "{}/{}".format(s["only_slide"], s["only_block"]),
            "{:.1f} {}".format(100 * d["delta"], ci(d["ci95"])),
            "{:.1f} {}".format(100 * lc["stratified_delta"], ci(lc["ci95"])),
            "{:.1f}/{:.1f}".format(lc["successes_slide"]["mean_steps"],
                                   lc["successes_block"]["mean_steps"]),
            "{:.0f}/{:.0f}".format(100 * sv["block_fail_share_if_slide_dirty"],
                                   100 * sv["block_fail_share_if_slide_clean"])]))
    table("regime",
          r"Paired regime statistics (\slideon{} minus \slideoff{}). SR: difference with "
          r"bootstrap 95\% interval, McNemar exact $p$, and episodes that succeed only under "
          r"\slideon{}/only under \slideoff{}. DSR: raw difference and the difference stratified "
          r"by episode-length quartile. Succ.\ len.: mean primitives of successful episodes "
          r"under \slideon{}/\slideoff{}. Fail: share (\%) of \slideon{} dirty/clean successes "
          r"that fail under \slideoff{}.",
          "tab:supp-regime", "lllllllr",
          r"Agent & $\Delta$SR & McNemar $p$ & Only & $\Delta$DSR & $\Delta$DSR strat. "
          r"& Succ.\ len. & Fail", rows, star=True, size="footnotesize", sep=3)


def signal_tables(intro):
    rows = []
    for bench in ("r2r", "rxr"):
        if bench == "rxr":
            rows.append(r"\midrule")
        for a, lab in AGENTS:
            x = intro[a].get(bench)
            if not x:
                continue
            S, F = x["signals_at_decision_steps"], x["signals_first_contact"]
            ent = "entropy_norm" if a in ("etpr1", "etpnav") else "entropy"

            def g(D, k):
                v = D.get(k, {}).get("3", {}).get("auroc")
                return r"\nmark" if v is None else "{:.3f}".format(v).lstrip("0")
            h = x["stall_history_forward"]
            rows.append(" & ".join([
                lab + (" (RxR)" if bench == "rxr" else ""),
                g(S, "geometry_inv"), g(F, "geometry_inv"), g(S, ent), g(F, ent),
                g(S, "stop_prob_inv"), g(F, "stop_prob_inv"), g(S, "margin_inv"),
                g(F, "margin_inv"), "{:,}".format(x["n_first_contact_decision_steps"]),
                "{:,}/{:,}".format(h["n_true_positive"], h["n_flagged"])]))
    table("first_contact",
          r"Collision-prediction AUROC (horizon of three primitives, \slideoff{}) on all "
          r"decisions (All) and on first-contact decisions only (1st: poses with no earlier "
          r"stall in the episode, where movement history is undefined by construction). "
          r"$n_1$: first-contact decisions. Hist.\ prec.: forward steps flagged by movement "
          r"history that collided, over all flagged.",
          "tab:supp-first-contact", "lcccccccccr",
          r"Agent & Depth & Depth 1st & Ent. & Ent. 1st & Stop & Stop 1st & Marg. & Marg. 1st "
          r"& $n_1$ & Hist.\ prec.", rows, star=True, size="footnotesize", sep=3)


def guard_tables(effect, mech):
    rows = []
    arms = ([(a, "r2r", "block") for a, _ in AGENTS[:5]] + [("etpr1", "rxr", "block"),
            ("etpnav", "rxr", "block")] + [(a, "r2r", "slide") for a in
                                           ("etpr1", "etpnav", "streamvln", "cma")])
    for a, b, r in arms:
        key = "{}/{}/{}{}".format(a, b, r, GUARD)
        e, m = effect[key], mech["guarded"][key]
        cl = m["sr_gain_by_unguarded_state"]["classes"]
        ns = m["sr_gain_by_unguarded_state"]["sr_change_on_not_stuck_episodes"]
        res = m["residual_collisions"]["share_by_source"]
        L = m["length"]
        bud = e.get("decision_budget")
        rows.append(" & ".join([
            "{}{} & {}".format(NAME[a], " (RxR)" if b == "rxr" else "",
                               r"\slideon{}" if r == "slide" else r"\slideoff{}"),
            "{}/{}".format(e["success_flips"]["gained"], e["success_flips"]["lost"]),
            pct(cl.get("stuck", {}).get("share_of_net_gain", 0), 0) if r == "block"
            else r"\nmark",
            "{:+.1f} {}".format(100 * ns["delta"], ci(ns["ci95"])),
            "{:.0f}/{:.0f}/{:.0f}".format(100 * res.get("maneuver", 0), 100 * res.get("pass", 0),
                                          100 * res.get("yield", 0)),
            pct(m["fragmentation"]["fragment_share_on"], 0),
            "{:.2f}".format(m["fragmentation"]["events_per_episode_on_merged"]),
            "{:.0f}/{:.0f}".format(L["steps_per_episode"]["off"], L["steps_per_episode"]["on"]),
            "{}/{}".format(pct(L["timeout_share"]["off"]), pct(L["timeout_share"]["on"])),
            "{}/{}".format(pct(bud["cap_hit_share_off"]), pct(bud["cap_hit_share_on"]))
            if bud else r"\nmark",
            "{:.2f}/{:.2f}".format(m["contact_per_metre"]["off"], m["contact_per_metre"]["on"])]))
        if (b, r) == ("r2r", "block") and a == "cma" or (b, r) == ("rxr", "block") and a == "etpnav":
            rows.append(r"\midrule")
    table("guard_mechanism",
          r"How the guard changes each run. Flips: failures turned into successes / successes "
          r"turned into failures. Stuck share: share of the net success gain from episodes "
          r"with a stuck event when unguarded (\slideoff{} only; for StreamVLN a further 19\% "
          r"comes from 500-step timeouts). Not stuck: success change "
          r"(points, 95\% interval) on episodes without a stuck event when unguarded. Residual: "
          r"remaining collision steps on escape forwards / unflagged policy forwards / yielded "
          r"forwards (\%). Frag.: guarded events within 12 primitives of the previous one "
          r"(\%); Merged: events per episode after merging them. Length, timeouts (500-step "
          r"limit, \%) and decision-budget exhaustion (ETP family, \%) are No guard/Guard, as "
          r"is contact per metre travelled.",
          "tab:supp-guard-mech", "llrrlcrrrrrr",
          r"Agent & Regime & Flips & Stuck & Not stuck & Residual & Frag. & Merged & Length "
          r"& Timeout & Budget & /m", rows, star=True, size="footnotesize", sep=2.5)

    rows = []
    base = "etpr1/r2r/block" + "_guard-"
    variants = [("No guard", None), ("stall only", "stall-only"), ("depth only (1 frame)", "depth-mem1"),
                (r"$\tau=0.02$\,m", "stall-mem1-tau0.02"), (r"\textbf{Guard}, $\tau=0.05$\,m", "stall-mem1"),
                (r"$\tau=0.10$\,m", "stall-mem1-tau0.10"), (r"$\tau=0.20$\,m", "stall-mem1-tau0.20"),
                ("8-frame depth", "stall"), ("8-frame depth, no stall", "default"),
                ("8-frame, no stall, no $K$", "uncapped"), ("oracle predictor", "oracle-mem1")]
    for lab, v in variants:
        e = effect[base + (v or "stall-mem1")]
        M = e["metrics"]
        side = "off" if v is None else "on"
        ev = mech["guarded"][base + (v or "stall-mem1")]["fragmentation"]
        own = "--" if v is None else pct(e["intervention"]["guard_owned_step_share"])
        rows.append(" & ".join([lab, pct(M["success"][side]), pct(M["spl"][side]),
                                pct(M["clean_success"][side]),
                                "{:.2f}".format(M["collision_count"][side]),
                                "{:.2f}".format(ev["events_per_episode_" + side]),
                                pct(M["stuck_flag"][side]), own]))
        if v is None or v == "stall-mem1-tau0.20" or v == "uncapped":
            rows.append(r"\midrule")
    rows.append(r"\multicolumn{8}{@{}l}{\textit{StreamVLN, R2R-CE, \slideoff{}}}\\")
    for lab, arm in (("No guard", "stall-mem1"), (r"\textbf{Guard}", "stall-mem1"),
                     ("oracle predictor", "oracle-mem1")):
        e = effect["streamvln/r2r/block_guard-" + arm]
        M = e["metrics"]
        side = "off" if lab == "No guard" else "on"
        ev = mech["guarded"]["streamvln/r2r/block_guard-" + arm]["fragmentation"]
        own = "--" if side == "off" else pct(e["intervention"]["guard_owned_step_share"])
        rows.append(" & ".join([lab, pct(M["success"][side]), pct(M["spl"][side]),
                                pct(M["clean_success"][side]),
                                "{:.2f}".format(M["collision_count"][side]),
                                "{:.2f}".format(ev["events_per_episode_" + side]),
                                pct(M["stuck_flag"][side]), own]))
    table("ablation_full",
          r"All guard variants on ETP-R1 (R2R-CE, \slideoff{}), and the oracle on StreamVLN. "
          r"SR, SPL, Clean (collision-free success), Stuck (proxy stuck event) and Own (share "
          r"of primitives chosen by the guard) in \%; collision steps and events per episode.",
          "tab:supp-ablation", "lrrrrrrr",
          r"Variant & SR & SPL & Clean & Steps & Events & Stuck & Own", rows, sep=3)


def memory_table(mem):
    rows = []
    for a in ("etpr1", "cma", "streamvln"):
        m = mem[a]["memory"]
        rows.append(" & ".join([NAME[a]] + ["{}/{}".format(pct(m["M%d_t0.05" % k]["precision"]),
                                                           pct(m["M%d_t0.05" % k]["recall"]))
                                            for k in (1, 2, 4, 8)]))
    table("memory",
          r"Depth-veto precision/recall (\%) when $k$ depth frames are accumulated, replayed "
          r"offline on 154 unguarded \slideoff{} episodes per agent ($\tau=0.05$\,m).",
          "tab:supp-memory", "lcccc",
          r"Agent & $k=1$ & $k=2$ & $k=4$ & $k=8$", rows, sep=4)


def runtime_table(rt):
    rows = []
    for k, lab in (("policy_ms", "Policy"), ("guard_ms", "Guard"), ("sim_ms", "Simulator"),
                   ("step_ms", "Whole step")):
        rows.append(" & ".join([lab, "{:.2f}".format(rt["off"]["per_step"][k]["mean"]),
                                "{:.2f}".format(rt["on"]["per_step"][k]["mean"])]))
    table("runtime",
          r"Mean wall-clock time per primitive (ms) for ETP-R1 on the first 50 R2R-CE "
          r"episodes under \slideoff{}, without and with the guard (CPU depth projection; "
          r"shared machine). The guard takes " + "{:.1f}".format(
              100 * rt["guard_share_of_step"]) + r"\% of a guarded step.",
          "tab:supp-runtime", "lrr", r"Component & No guard & Guard", rows, sep=6)


def main():
    os.makedirs(OUT, exist_ok=True)
    p1_tables(load("paper_numbers.json"), load("regime_stats.json"))
    signal_tables(load("introspection.json"))
    guard_tables(load("guard_effect.json"), load("guard_mechanism.json"))
    memory_table(load("guard_memory.json"))
    runtime_table(load("runtime.json"))
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
