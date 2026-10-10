#!/usr/bin/env python3
"""results/summary/*.json -> outputs/tables/generated/numbers_macros.tex.

What: every number the prose quotes, as `\\pn{dotted.key}` (defined in
paper/src/00_preamble.tex). A key never generated renders as `\\TBD`, so a sentence about a
run not yet done shows a red cell instead of a stale or typed number. Keys:

  <agent>.<bench>.<regime>.{sr,dsr,dsrlo,dsrhi,nc,stuck,cpds,...}   P1/P2, paper_numbers.json
  <agent>.<bench>.{dsr_pp,ddsr_pp,kmiss}                           regime delta
  <agent>.<bench>.auc.{geom,hist,entnorm,ent,stop,marg}            introspection.json, k = 3
  <agent>.<bench>.hist.{recall,flagged,tp}                         stall history on forwards
  <agent>.<bench>.{swap,avail}{10,50}                              headroom.json
  <agent>.<bench>.<regime>.<guard>.<metric>.{off,on,d,lo,hi,rel}   guard_effect via paper_numbers
  <agent>.<bench>.<regime>.<guard>.{stalls,aborts,owned,gained,lost,...}
  <agent>.mem.{M1,M8}.{prec,recall,false}                          guard_memory.json
  range.*                                                          min--max over agents

Usage: analysis/prose_numbers.py      (after analysis/make_tables.py; `make tables` runs both)
Modern Python (3.9+); never imported by the harness.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import (AGENT_META, GUARD_SET, METHOD, ORDER, PLANNED, TBD, _load,  # noqa: E402
                         frac3, get, num, pct, snum, spct, write)


def make_macros(numbers, intro, headroom, memory):
    """`\\pn{key}` values for the prose. Keys are dotted paths; a key never generated
    renders as \\TBD (src/00_preamble.tex)."""
    mac = {}

    def put(key, val):
        if val is not None and val != TBD:
            mac[key] = val

    for aid in list(AGENT_META) + list(PLANNED):
        for bench in ("r2r", "rxr"):
            for reg in ("block", "slide"):
                x = get(numbers, aid, bench, reg)
                if not x:
                    continue
                k = "{}.{}.{}.".format(aid, bench, reg)
                p1 = x["p1"]
                put(k + "sr", pct(x["n_success"] / x["n_episodes"]))
                put(k + "dsr", pct(p1["dirty_success_rate"]))
                put(k + "dsrlo", pct(p1["dirty_success_ci95"][0]))
                put(k + "dsrhi", pct(p1["dirty_success_ci95"][1]))
                put(k + "nc", pct(p1["nc_rate_mean"]))
                put(k + "stuck", pct(p1["stuck_rate"]))
                put(k + "cpds", num(p1["collisions_per_dirty_success"]))
                put(k + "cpdsmed", num(p1["collisions_per_dirty_success_median"], 0))
                put(k + "shareinsucc", pct(p1["share_of_collisions_in_successes"]))
                put(k + "dirtysuccess", pct(x["joint_rates"]["dirty_success"]))
                put(k + "cleansuccess", pct(x["joint_rates"]["clean_success"]))
                put(k + "ncsucc", pct(p1["nc_rate_success"]["mean"]))
                put(k + "ncfail", pct(p1["nc_rate_failure"]["mean"]))
                put(k + "tir", pct(p1["time_in_risk_rate_mean"]))
                put(k + "proxydsr", pct(x["p2"]["dirty_success_rate_proxy"]))
            d = get(numbers, aid, bench, "regime_delta")
            if d:
                k = "{}.{}.".format(aid, bench)
                put(k + "dsr_pp", snum(d["delta_sr_pp"]))
                put(k + "ddsr_pp", snum(d["delta_dirty_success_pp"]))
                put(k + "kmiss", pct(d["kappa_miss"]))
            s = get(intro, aid, bench)
            if s:
                k = "{}.{}.".format(aid, bench)
                for sig, short in (("geometry_inv", "geom"), ("stall_history", "hist"),
                                   ("entropy_norm", "entnorm"), ("entropy", "ent"),
                                   ("stop_prob_inv", "stop"), ("margin_inv", "marg")):
                    put(k + "auc." + short, frac3(get(s, "signals_at_decision_steps", sig, "3", "auroc")))
                f = s["stall_history_forward"]
                put(k + "hist.recall", pct(f["recall"]))
                put(k + "hist.flagged", "{:,}".format(f["n_flagged"]).replace(",", "{,}"))
                put(k + "hist.tp", "{:,}".format(f["n_true_positive"]).replace(",", "{,}"))
            h = get(headroom, aid, bench, "eps_sweep")
            if h:
                for eps, tag in (("0.1", "10"), ("0.5", "50")):
                    put("{}.{}.swap{}".format(aid, bench, tag), pct(h[eps]["headroom_rate_unconditional"]))
                    put("{}.{}.avail{}".format(aid, bench, tag), pct(h[eps]["availability_rate_among_risky"]))
            # block_tryout: the guard's veto inside Tryout, paired against Tryout
            for reg in ("block", "slide", "block_tryout"):
                for name, g in (get(numbers, aid, bench, "guard", reg) or {}).items():
                    k = "{}.{}.{}.{}.".format(aid, bench, reg, name)
                    put_pair(put, k, g["metrics"])
                    iv = g["intervention"]
                    put(k + "stalls", num(iv["stall_vetoes_per_episode"]))
                    put(k + "aborts", num(iv["aborts_per_episode"]))
                    put(k + "yields", num(iv["yields_per_episode"], 2))
                    put(k + "abortep", pct(iv["episodes_with_abort"], 0))
                    put(k + "owned", pct(iv["guard_owned_step_share"]))
                    put(k + "gained", str(g["success_flips"]["gained"]))
                    put(k + "lost", str(g["success_flips"]["lost"]))
                    r = g["residual_collisions_by_guard_action"]
                    tot = float(sum(r.values()))
                    if tot:
                        put(k + "resid.maneuver", pct(r.get("maneuver", 0) / tot, 0))
                        put(k + "resid.pass", pct(r.get("pass", 0) / tot, 0))
                    b = g.get("decision_budget")
                    if b:
                        put(k + "caphit.off", pct(b["cap_hit_share_off"]))
                        put(k + "caphit.on", pct(b["cap_hit_share_on"]))
                        put(k + "overcap", str(b["episodes_over_allowed_cap_on"]))
            base = get(numbers, aid, bench, "baseline")
            for name, arm in (base or {}).items():
                for m, v in arm["episode_means"].items():
                    put("{}.{}.{}.{}".format(aid, bench, name, m),
                        pct(v) if m in ("success", "spl", "ndtw", "stuck_flag") else num(v))
                # paired comparisons (analysis/guard_effect.py): <arm>.vsoff.<metric>.*
                # = baseline minus unguarded; <arm>.vsguard.<metric>.* = guard minus
                # baseline; <arm>.combovsguard.<metric>.* = baseline + guard veto minus guard
                for tag, key in (("vsoff", "vs_off"), ("vsguard", "guard_vs_baseline"),
                                 ("combovsguard", "combo_vs_guard")):
                    pair = (arm.get("paired") or {}).get(key)
                    if pair:
                        k = "{}.{}.{}.{}.".format(aid, bench, name, tag)
                        put_pair(put, k, pair["metrics"])
                        put(k + "gained", str(pair["success_flips"]["gained"]))
                        put(k + "lost", str(pair["success_flips"]["lost"]))

    for aid, arm in memory.items():
        mem = arm["memory"]
        for key in ("M1_t0.05", "M8_t0.05"):
            k = "{}.mem.{}.".format(aid, key.split("_")[0])
            put(k + "prec", pct(mem[key]["precision"]))
            put(k + "recall", pct(mem[key]["recall"]))
            put(k + "false", num(mem[key]["false_vetoes_per_episode"], 2))
        put("{}.mem.episodes".format(aid), str(arm["episodes"]))

    make_ranges(put, numbers, intro, headroom)
    lines = [r"\expandafter\def\csname pn@{}\endcsname{{{}}}".format(k, v) for k, v in sorted(mac.items())]
    write("numbers_macros", "\n".join(lines) + "\n")
    return len(mac)


def put_pair(put, k, metrics):
    """`<k><metric>.{off,on,d,lo,hi,rel}` for one paired comparison's metrics block."""
    for m, v in metrics.items():
        f, sf = ((pct, spct) if m in ("success", "spl", "ndtw", "nc_rate",
                                      "dirty_success_flag", "stuck_flag", "clean_success")
                 else (num, snum))
        nd = 2 if m == "collision_event_count" else 1
        put(k + m + ".off", f(v["off"], nd))
        put(k + m + ".on", f(v["on"], nd))
        put(k + m + ".d", sf(v["delta"], nd))
        put(k + m + ".lo", sf(v["ci95"][0], nd))
        put(k + m + ".hi", sf(v["ci95"][1], nd))
        if v["off"]:
            put(k + m + ".rel", snum(100.0 * v["delta"] / v["off"], 0))


def _span(vals, f):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    lo, hi = f(min(vals)), f(max(vals))
    return lo if lo == hi else "{}--{}".format(lo, hi)


def make_ranges(put, numbers, intro, headroom):
    """`range.*` keys: min--max over the agents that have the quantity, so a sentence
    about "every agent" stays true as rows land."""
    landed = [a for a in ORDER if get(numbers, a, "r2r", "block") and get(numbers, a, "r2r", "slide")]
    put("range.nagents", str(len(landed)))
    for reg in ("block", "slide"):
        put("range.dsr." + reg, _span([get(numbers, a, "r2r", reg, "p1", "dirty_success_rate")
                                        for a in landed], pct))
        put("range.stuck." + reg, _span([get(numbers, a, "r2r", reg, "p1", "stuck_rate")
                                          for a in landed], pct))
        put("range.nc." + reg, _span([get(numbers, a, "r2r", reg, "p1", "nc_rate_mean")
                                       for a in landed], pct))
    put("range.dsr_pp", _span([get(numbers, a, "r2r", "regime_delta", "delta_sr_pp") for a in landed],
                              lambda x: "{:.1f}".format(x)))
    put("range.kmiss", _span([get(numbers, a, b, "regime_delta", "kappa_miss")
                              for a in ORDER for b in ("r2r", "rxr")], pct))
    sig = lambda a, b, n: get(intro, a, b, "signals_at_decision_steps", n, "3", "auroc")
    policy = [sig(a, b, n) for a in ORDER for b in ("r2r", "rxr")
              for n in ("entropy_norm", "entropy", "stop_prob_inv", "margin_inv")]
    put("range.auc.policymax", frac3(max(v for v in policy if v is not None)))
    put("range.auc.policymin", frac3(min(v for v in policy if v is not None)))
    for n, short in (("geometry_inv", "geom"), ("stall_history", "hist")):
        put("range.auc.r2r." + short, _span([sig(a, "r2r", n) for a in ORDER], frac3))
    put("range.hist.recall.r2r", _span([get(intro, a, "r2r", "stall_history_forward", "recall")
                                        for a in ORDER], pct))
    put("range.hist.recall", _span([get(intro, a, b, "stall_history_forward", "recall")
                                    for a in ORDER for b in ("r2r", "rxr")], pct))
    put("range.swap10.r2r", _span([get(headroom, a, "r2r", "eps_sweep", "0.1", "headroom_rate_unconditional")
                                   for a in ORDER], pct))
    guarded = [get(numbers, a, "r2r", "guard", "block", METHOD) for a in GUARD_SET]
    guarded = [g for g in guarded if g]
    put("range.guard.n", str(len(guarded)))
    for m in ("success", "spl"):
        put("range.guard.{}.d".format(m), _span([g["metrics"][m]["delta"] for g in guarded],
                                                lambda x: "{:.1f}".format(100 * x)))
    put("range.guard.coll.rel", _span([-g["metrics"]["collision_count"]["delta"] / g["metrics"]["collision_count"]["off"]
                                       for g in guarded], lambda x: "{:.0f}".format(100 * x)))
    put("range.guard.stuck.off", _span([g["metrics"]["stuck_flag"]["off"] for g in guarded], pct))
    put("range.guard.stuck.on", _span([g["metrics"]["stuck_flag"]["on"] for g in guarded], pct))
    put("range.guard.owned", _span([g["intervention"]["guard_owned_step_share"] for g in guarded], pct))
    recov = []
    for a in GUARD_SET:
        g = get(numbers, a, "r2r", "guard", "block", METHOD)
        sl = get(numbers, a, "r2r", "slide")
        if g and sl:
            gap = sl["n_success"] / sl["n_episodes"] - g["metrics"]["success"]["off"]
            recov.append(g["metrics"]["success"]["delta"] / gap if gap > 0 else None)
    put("range.guard.recovered", _span(recov, lambda x: "{:.0f}".format(100 * x)))




def main():
    n = make_macros(_load("paper_numbers.json"), _load("introspection.json"),
                    _load("headroom.json"), _load("guard_memory.json"))
    print("wrote numbers_macros.tex ({} keys)".format(n))


if __name__ == "__main__":
    main()
