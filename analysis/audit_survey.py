#!/usr/bin/env python3
"""Derive every audit number the paper quotes from survey/papers.csv, and emit the LaTeX
for the audit tables so the tables and the prose can never drift from the coding sheet.
Run after editing the CSV.

Writes outputs/tables/generated/{audit_table_a,audit_table_b,audit_family_table,
audit_denominators,audit_macros,reported_sr_table}.tex and prints the stats JSON.

Modern Python; never imported by the harness.

Ported verbatim from vln_research/pipeline/analysis/audit_stats.py — only the input/output
paths change (survey/papers.csv, survey/reported_sr_r2rce.csv, outputs/tables/generated/).
"""
import csv
import json
import pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
CSV = REPO / "survey" / "papers.csv"
TEXDIR = REPO / "outputs" / "tables" / "generated"
TEXDIR.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(CSV.open()))


def sel(**kw):
    out = rows
    for k, v in kw.items():
        vals = v if isinstance(v, (list, tuple)) else [v]
        out = [r for r in out if r[k] in vals]
    return out


contact = sel(contact_possible="Y")
comparable = [r for r in contact if r["std_vlnce_bench"] == "Y"]

stats = {
    "n_audited": len(rows),
    "n_contact_possible": len(contact),
    "n_contact_impossible": len(sel(contact_possible="N")),
    "n_reports_safety_among_contact": sum(r["reports_safety_metric"] == "Y" for r in contact),
    "n_comparable": len(comparable),
    "n_reports_safety_among_comparable": sum(r["reports_safety_metric"] == "Y" for r in comparable),
    "n_regime_stated_among_comparable": sum(r["physics_regime_stated"] == "Y" for r in comparable),
    "n_outcome_conditioned": sum(r["outcome_conditioned"] == "Y" for r in rows),
    "n_paired_stats": sum(r["paired_stats"] == "Y" for r in rows),
    "collision_definitions": dict(
        Counter(r["collision_definition"] for r in contact if r["reports_safety_metric"] == "Y")
    ),
    "safety_metric_names": sorted(
        {r["safety_metric"] for r in contact if r["reports_safety_metric"] == "Y"}
    ),
    "families": dict(Counter(r["family"] for r in rows)),
    "verification_levels": dict(Counter(r["verification"] for r in rows)),
    "year_range": [min(int(r["year"]) for r in rows), max(int(r["year"]) for r in rows)],
}

cell = sel(proactive="Y", training_free="Y", plan_aware="Y", std_vlnce_bench="Y")
stats["empty_cell_members"] = [r["key"] for r in cell]
stats["n_empty_cell"] = len(cell)

axes = ["proactive", "training_free", "plan_aware", "std_vlnce_bench"]
near = []
for r in contact:
    missing = [a for a in axes if r[a] != "Y"]
    if len(missing) == 1:
        near.append({"key": r["key"], "short_name": r["short_name"], "misses": missing[0]})
stats["one_axis_short"] = near

(TEXDIR / "audit_stats.json").write_text(json.dumps(stats, indent=2))


def esc(s):
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


MARK = {"Y": r"\cmark", "N": r"\xmark", "na": r"\nmark", "unk": "?"}
FAMILY_ORDER = [
    ("discrete", "Discrete (nav-graph) VLN"),
    ("metric", "VLN evaluation metrics"),
    ("fm-vln", "Foundation-model / zero-shot VLN"),
    ("vlnce", "VLN in continuous environments"),
    ("nav-vla", "Navigation VLAs"),
    ("analysis", "VLN diagnosis \\& analysis"),
    ("robustness", "VLN robustness"),
    ("uncertainty", "Uncertainty \\& failure anticipation"),
    ("vln-safety", "Dedicated VLN safety"),
    ("benchmark", "Safety-aware benchmarks"),
    ("adversarial", "Adversarial VLN"),
    ("robot-safety", "Embodied / robot safety filters"),
    ("survey", "Surveys"),
]

lines = []
for fam, title in FAMILY_ORDER:
    frs = [r for r in rows if r["family"] == fam]
    if not frs:
        continue
    frs.sort(key=lambda r: (int(r["year"]), r["short_name"]))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{@{}l}{\textit{%s}}\\[1pt]" % title)
    for r in frs:
        dagger = r"$^{\dagger}$" if r["verification"] == "abstract" else ""
        lines.append(
            " & ".join([
                r"\hspace{0.6em}\citet{%s}%s" % (r["key"], dagger),
                r["year"],
                MARK.get(r["contact_possible"], "?"),
                MARK.get(r["reports_safety_metric"], "?"),
                MARK.get(r["physics_regime_stated"], "?"),
                MARK.get(r["outcome_conditioned"], "?"),
                MARK.get(r["proactive"], "?"),
                MARK.get(r["training_free"], "?"),
            ]) + r"\\"
        )
AUDIT_COLS = "\\toprule\nPaper & Year & C & S & R & O & P & T \\\\"


def _float(body, label, caption):
    return "\n".join([
        r"\begin{table*}[p]",
        r"\centering\footnotesize",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\caption{%s}" % caption,
        r"\label{%s}" % label,
        r"\begin{tabular}{@{}llcccccc@{}}",
        AUDIT_COLS,
        body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]) + "\n"


_bounds = [i for i, ln in enumerate(lines) if ln.startswith(r"\midrule")]
_split = min(_bounds, key=lambda i: abs(i - len(lines) // 2))
(TEXDIR / "audit_table_a.tex").write_text(_float(
    "\n".join(lines[:_split]), "tab:audit-full",
    r"Reporting audit, part 1 of 2 (\AuditYearLo{}--\AuditYearHi{}). "
    r"Columns are defined in Appendix~\ref{app:audit}."))
(TEXDIR / "audit_table_b.tex").write_text(_float(
    "\n".join(lines[_split:]), "tab:audit-full-b",
    r"Reporting audit, part 2 of 2. Columns as in Table~\ref{tab:audit-full}."))

fam_lines = []
for fam, title in FAMILY_ORDER:
    frs = [r for r in rows if r["family"] == fam]
    if not frs:
        continue
    n = len(frs)
    nc = sum(r["contact_possible"] == "Y" for r in frs)
    ns = sum(r["reports_safety_metric"] == "Y" for r in frs)
    nr = sum(r["physics_regime_stated"] == "Y" for r in frs)
    yrs = sorted(int(r["year"]) for r in frs)
    span = "%d--%d" % (yrs[0], yrs[-1]) if yrs[0] != yrs[-1] else "%d" % yrs[0]
    fam_lines.append("%s & %s & %d & %d & %d & %d" % (title, span, n, nc, ns, nr) + r"\\")
fam_lines.append(r"\midrule")
fam_lines.append(
    r"\textbf{Total} & \textbf{%d--%d} & \textbf{%d} & \textbf{%d} & \textbf{%d} & \textbf{%d}\\"
    % (stats["year_range"][0], stats["year_range"][1], stats["n_audited"],
       stats["n_contact_possible"], stats["n_reports_safety_among_contact"]
       + sum(r["reports_safety_metric"] == "Y" and r["contact_possible"] != "Y" for r in rows),
       sum(r["physics_regime_stated"] == "Y" for r in rows)))
(TEXDIR / "audit_family_table.tex").write_text("\n".join([
    r"\begin{table}[t]",
    r"\centering\small",
    r"\setlength{\tabcolsep}{3pt}",
    r"\caption{Breadth of the screen. \emph{Contact}: the evaluation setting admits"
    r" physical collision. \emph{Safety}: reports any safety or collision metric."
    r" \emph{Regime}: states the simulator collision-handling regime. Every family"
    r" scores zero on conditioning safety statistics on episode outcome, so that"
    r" column is omitted.}",
    r"\label{tab:audit-summary}",
    r"\begin{tabular}{@{}lccccc@{}}",
    r"\toprule",
    r"Family & Years & $n$ & Contact & Safety & Regime \\",
] + fam_lines + [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]) + "\n")

_regime_contact = sum(r["physics_regime_stated"] == "Y" for r in contact)
_safety_all = sum(r["reports_safety_metric"] == "Y" for r in rows)
_regime_all = sum(r["physics_regime_stated"] == "Y" for r in rows)
den_lines = [
    r"All surveyed papers & %d & %d & %d & %d\\"
    % (stats["n_audited"], _safety_all, _regime_all, stats["n_outcome_conditioned"]),
    r"\quad evaluation admits contact & %d & %d & %d & %d\\"
    % (stats["n_contact_possible"], stats["n_reports_safety_among_contact"],
       _regime_contact, stats["n_outcome_conditioned"]),
    r"\quad\quad on standard VLN-CE benchmarks & %d & %d & %d & %d\\"
    % (stats["n_comparable"], stats["n_reports_safety_among_comparable"],
       stats["n_regime_stated_among_comparable"], stats["n_outcome_conditioned"]),
]
(TEXDIR / "audit_denominators.tex").write_text("\n".join([
    r"\begin{table}[t]",
    r"\centering\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\caption{What the surveyed literature reports about physical safety, by"
    r" denominator. Counts are papers. The third row is the set whose numbers a"
    r" VLN reader would place side by side.}",
    r"\label{tab:reporting}",
    r"\begin{tabular}{@{}lcccc@{}}",
    r"\toprule",
    r" & & \multicolumn{3}{c}{of which, report\dots} \\",
    r"\cmidrule(lr){3-5}",
    r"Denominator & $n$ & a safety & the & safety by \\",
    r" & & metric & regime & outcome \\",
    r"\midrule",
] + den_lines + [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]) + "\n")

macros = {
    "AuditN": stats["n_audited"],
    "AuditContact": stats["n_contact_possible"],
    "AuditContactSafety": stats["n_reports_safety_among_contact"],
    "AuditComparable": stats["n_comparable"],
    "AuditComparableSafety": stats["n_reports_safety_among_comparable"],
    "AuditRegimeStated": stats["n_regime_stated_among_comparable"],
    "AuditOutcomeCond": stats["n_outcome_conditioned"],
    "AuditPaired": stats["n_paired_stats"],
    "AuditEmptyCell": stats["n_empty_cell"],
    "AuditYearLo": stats["year_range"][0],
    "AuditYearHi": stats["year_range"][1],
    "AuditNDefs": len(stats["collision_definitions"]),
}
(TEXDIR / "audit_macros.tex").write_text(
    "\n".join(r"\newcommand{\%s}{%s}" % (k, v) for k, v in macros.items()) + "\n"
)

sr_rows = list(csv.DictReader((REPO / "survey" / "reported_sr_r2rce.csv").open()))
sr_body = [
    " & ".join([r"\citet{%s}" % r["key"], r["year"], r["sr_reported"],
                r["spl_reported"], r"\cmark" if r["regime_stated"] == "Y" else r"\xmark"])
    + r"\\"
    for r in sr_rows
]
(TEXDIR / "reported_sr_table.tex").write_text("\n".join([
    r"\begin{table}[t]",
    r"\centering\small",
    r"\setlength{\tabcolsep}{4pt}",
    r"\caption{Published R2R-CE validation-unseen results for %d widely compared"
    r" agents, and whether the paper states the collision-handling regime the"
    r" numbers depend on. Table~\ref{tab:regime} measures the shift on each"
    r" agent of our model set; on the two measured so far it is 15--19 points,"
    r" larger than any single step in this table.}" % len(sr_rows),
    r"\label{tab:comparability}",
    r"\begin{tabular}{@{}llccc@{}}",
    r"\toprule",
    r"Agent & Year & SR & SPL & Regime stated \\",
    r"\midrule",
] + sr_body + [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]) + "\n")

if __name__ == "__main__":
    print(json.dumps(stats, indent=2))
