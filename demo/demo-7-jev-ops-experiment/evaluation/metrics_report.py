"""Markdown metrics report: the standard figures people ask for in reviews.

Every number comes from evaluation.metrics.compute() over the run records.
Columns are engines (one result file each).
"""
from __future__ import annotations

from evaluation.metrics import INVALID, LABELS, compute


def _p(x, digits=0) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _ci(ci) -> str:
    return "" if ci is None else f" [{ci[0] * 100:.0f}–{ci[1] * 100:.0f}%]"


def _f(x, spec=".2f") -> str:
    return "n/a" if x is None else format(x, spec)


def _ms(x) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def _usd(x, digits=4) -> str:
    return "n/a" if x is None else f"${x:.{digits}f}"


def _table(labels, rows) -> list[str]:
    out = ["| | " + " | ".join(labels) + " |", "|" + "---|" * (len(labels) + 1)]
    out += [f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows]
    return out + [""]


def render_metrics(runs: list[tuple[str, list[dict]]], labels: list[str],
                   warnings: list[str]) -> str:
    ms = [compute(rs) for _, rs in runs]
    col = lambda f: [f(m) for m in ms]  # noqa: E731
    out = ["# Metrics report", "",
           "Standard metrics for the Tier 1 decision benchmark, computed from the run records. "
           "Definitions are in the glossary at the end. Square brackets are 95% confidence "
           "intervals (Wilson). Errored runs (no answer) are left out of quality metrics; "
           "invalid answers count as wrong.", ""]
    out += warnings
    out += ["Sources: " + ", ".join(f"`{src}` ({lab})" for lab, (src, _) in zip(labels, runs)), ""]
    small = [lab for lab, m in zip(labels, ms) if m["scored"] < 30]
    if small:
        out += [f"> **Small sample:** {', '.join(small)} has fewer than 30 scored runs. "
                "Read the confidence intervals, not just the point values.", ""]

    out += ["## Headline", ""]
    out += _table(labels, [
        ("Accuracy (operational decision)", col(lambda m: _p(m["accuracy"]) + _ci(m["accuracy_ci"]))),
        ("Macro F1: severity", col(lambda m: _f(m["fields"]["severity"]["macro_f1"]))),
        ("Macro F1: action", col(lambda m: _f(m["fields"]["action"]["macro_f1"]))),
        ("F1: human review", col(lambda m: _f(m["human_review"]["f1"]))),
        ("Incident detection recall", col(lambda m: _p(m["detection"]["recall"]))),
        ("Incident false positive rate", col(lambda m: _p(m["detection"]["false_positive_rate"]))),
        ("Latency p95", col(lambda m: _ms(m["latency"]["p95"]) + " ms")),
        ("Availability", col(lambda m: _p(m["reliability"]["availability"], 1))),
        ("Cost per run", col(lambda m: _usd(m["cost"]["per_run"]))),
    ])

    out += ["## Decision quality by field", "",
            "Accuracy is the share of answers that match the expected value. Macro F1 averages "
            "F1 over the classes, so rare classes count as much as common ones. Weighted F1 "
            "weights each class by how often it is expected.", ""]
    rows = []
    for f in LABELS:
        note = " (diagnosis, not in PASS/FAIL)" if f == "probable_cause" else ""
        rows += [(f"{f}{note}: accuracy", col(lambda m, f=f: _p(m["fields"][f]["accuracy"])
                                               + _ci(m["fields"][f]["accuracy_ci"]))),
                 (f"{f}: macro F1", col(lambda m, f=f: _f(m["fields"][f]["macro_f1"]))),
                 (f"{f}: weighted F1", col(lambda m, f=f: _f(m["fields"][f]["weighted_f1"])))]
    out += _table(labels, rows)

    out += ["## Per-class precision, recall and F1", "",
            "Each cell is `F1 (precision / recall, n expected)`. `n/a` means the class was "
            "never expected and never predicted by that engine.", ""]
    for f, classes in LABELS.items():
        out += [f"### {f}", ""]
        rows = []
        for c in list(classes) + [INVALID]:
            cells = []
            for m in ms:
                k = m["fields"][f]["classes"].get(c)
                cells.append("n/a" if k is None else
                             f"{_f(k['f1'])} ({_f(k['precision'])} / {_f(k['recall'])}, n={k['support']})")
            if any(x != "n/a" for x in cells):
                rows.append((c, cells))
        out += _table(labels, rows)

    out += ["## Incident detection", "",
            "Treats the decision as a detector: an incident is any expected severity other than "
            "`normal`, and the engine detects one when it answers a severity other than `normal`. "
            "An invalid answer counts as not detected. Only the `healthy` scenario is a "
            "negative, so the false positive rate rests on few runs.", ""]
    d = lambda k: col(lambda m: str(m["detection"][k]))  # noqa: E731
    out += _table(labels, [
        ("True positives", d("tp")), ("False positives", d("fp")),
        ("False negatives (missed)", d("fn")), ("True negatives", d("tn")),
        ("Precision", col(lambda m: _p(m["detection"]["precision"]))),
        ("Recall (detection rate)", col(lambda m: _p(m["detection"]["recall"]) + _ci(m["detection"]["recall_ci"]))),
        ("F1", col(lambda m: _f(m["detection"]["f1"]))),
        ("False positive rate", col(lambda m: _p(m["detection"]["false_positive_rate"])
                                    + _ci(m["detection"]["fpr_ci"]))),
        ("Miss rate (false negative rate)", col(lambda m: _p(m["detection"]["miss_rate"]))),
        ("Specificity", col(lambda m: _p(m["detection"]["specificity"]))),
    ])

    out += ["## Human review (yes is positive)", ""]
    out += _table(labels, [
        ("TP / FP / FN / TN", col(lambda m: " / ".join(str(m["human_review"][k]) for k in ("tp", "fp", "fn", "tn")))),
        ("Precision", col(lambda m: _p(m["human_review"]["precision"]))),
        ("Recall", col(lambda m: _p(m["human_review"]["recall"]))),
        ("F1", col(lambda m: _f(m["human_review"]["f1"]))),
        ("False positive rate (asks for review when none is needed)",
         col(lambda m: _p(m["human_review"]["false_positive_rate"]))),
    ])

    out += ["## Operational risk", "",
            "How the wrong answers go wrong. Over-reacting wastes on-call time; under-reacting "
            "lets an incident run.", ""]
    out += _table(labels, [
        ("Severity over-estimated", col(lambda m: f"{m['severity_bias']['over']} ({_p(m['severity_bias']['over_rate'])})")),
        ("Severity under-estimated", col(lambda m: f"{m['severity_bias']['under']} ({_p(m['severity_bias']['under_rate'])})")),
        ("Unwarranted disruptive action (restart or escalate when not expected)",
         col(lambda m: f"{m['action_risk']['unwarranted_disruptive']}/{m['action_risk']['calm_runs']} "
                       f"({_p(m['action_risk']['unwarranted_rate'])})")),
        ("Missed escalation", col(lambda m: f"{m['action_risk']['missed_escalation']}/{m['action_risk']['escalation_runs']} "
                                            f"({_p(m['action_risk']['missed_escalation_rate'])})")),
    ])

    out += ["## Calibration", "",
            "Brier score of the engine's probability that a human should review, against the "
            "expected answer. Lower is better; 0.25 is what always answering 50% scores. Only "
            "engines that return a probability (Jev) have one.", ""]
    out += _table(labels, [
        ("Brier score (human review)", col(lambda m: _f(m["calibration"]["brier"], ".3f"))),
        ("Runs with a probability", col(lambda m: str(m["calibration"]["n"]))),
    ])

    out += ["## Latency (round trip per decision)", ""]
    out += _table(labels, [(k, col(lambda m, k=k: _ms(m["latency"][k]) + " ms"))
                           for k in ("mean", "p50", "p90", "p95", "p99", "max")]
                  + [("Runs measured", col(lambda m: str(m["latency"]["n"])))])

    out += ["## Reliability", ""]
    out += _table(labels, [
        ("Runs", col(lambda m: str(m["reliability"]["runs"]))),
        ("Availability (got an answer)", col(lambda m: _p(m["reliability"]["availability"], 1))),
        ("Error rate (no answer)", col(lambda m: _p(m["reliability"]["error_rate"], 1))),
        ("Invalid answer rate", col(lambda m: _p(m["reliability"]["invalid_rate"], 1))),
        ("Runs that needed a retry", col(lambda m: _p(m["reliability"]["retry_rate"], 1))),
    ])

    out += ["## Cost and tokens", ""]
    out += _table(labels, [
        ("Tokens in per run", col(lambda m: _ms(m["cost"]["tokens_in"]))),
        ("Tokens out per run", col(lambda m: _ms(m["cost"]["tokens_out"]))),
        ("Cost per run", col(lambda m: _usd(m["cost"]["per_run"]))),
        ("Cost per correct decision", col(lambda m: _usd(m["cost"]["per_correct"]))),
        ("Total cost", col(lambda m: _usd(m["cost"]["total"]))),
        ("Cost source", col(lambda m: ", ".join(m["cost"]["sources"]) or "none")),
    ])
    out += ["Cost is `n/a` unless the engine reports one or prices are set in `.env`. "
            "None is invented. `reported:claude-cli` includes the CLI's prompt-caching premium.", ""]

    out += ["## Not applicable to this benchmark", "",
            "| Metric | Why it is not reported |", "|---|---|",
            "| MTTD (mean time to detect) | Needs a fault that starts at a known time and a "
            "detector that watches over time. Tier 1 gives the engine one snapshot per run, so "
            "there is nothing to time. Tier 2 measures it as `detection_seconds` (fault "
            "manifestation to decision) once Tier 2 runs exist. Decision latency above is not MTTD. |",
            "| MTTR (mean time to repair) | Nothing is repaired. The benchmark records the "
            "recommended action and never carries it out. |",
            "| MTBF (mean time between failures) | Failures are injected by the experiment, so "
            "their spacing says nothing about a system. |",
            "| Uptime, SLO attainment | There is no running service in Tier 1. Availability "
            "above is the engine's API answering, not uptime. |",
            "| ROC AUC, PR AUC | Needs a score per run for every class. Only Jev returns "
            "probabilities, so they cannot be compared across engines. |", ""]

    out += ["## Appendix: confusion matrices", "",
            "Rows are the expected value, columns what the engine answered.", ""]
    for lab, m in zip(labels, ms):
        out += [f"### {lab}", ""]
        for f, classes in LABELS.items():
            conf = m["fields"][f]["confusion"]
            cols = [c for c in list(classes) + [INVALID]
                    if c in conf or any(c in row for row in conf.values())]
            if not conf:
                continue
            out += [f"**{f}**", "", "| expected \\ answered | " + " | ".join(cols) + " |",
                    "|" + "---|" * (len(cols) + 1)]
            for exp in classes:
                if exp in conf:
                    out.append(f"| {exp} | " + " | ".join(str(conf[exp].get(c, 0)) for c in cols) + " |")
            out.append("")

    out += ["## Glossary", "",
            "- **Accuracy**: correct answers divided by scored runs. For the operational decision, "
            "a run is correct only when severity, action and human review are all right.",
            "- **Precision** (for one class): of the times the engine answered this class, the "
            "share that were right. Low precision means false alarms.",
            "- **Recall** (for one class): of the times this class was expected, the share the "
            "engine got. Low recall means misses. For detection this is the detection rate.",
            "- **F1**: the harmonic mean of precision and recall, 2PR/(P+R). It is high only "
            "when both are.",
            "- **Macro F1**: the plain average of per-class F1 over every class that was expected "
            "or answered. Every class counts the same, so it exposes weak handling of rare "
            "classes. A class the engine answered but that was never expected scores 0 and pulls "
            "the average down (the scikit-learn convention).",
            "- **Weighted F1**: per-class F1 averaged by how often each class is expected. Closer "
            "to accuracy.",
            "- **False positive rate**: of the runs where nothing was wrong (or no review was "
            "needed), the share where the engine said otherwise.",
            "- **Miss rate**: of the runs with an incident, the share the engine did not flag. "
            "Equal to 1 minus recall.",
            "- **Specificity**: 1 minus the false positive rate.",
            "- **Confidence interval [a–b%]**: the range the true rate is likely in (95%, Wilson "
            "score). Wide intervals mean too few runs to be sure.",
            "- **p50, p90, p95, p99**: the latency below which 50%, 90%, 95% or 99% of decisions "
            "came back. p95 is the usual \"worst normal case\" figure; mean hides slow outliers.",
            "- **Availability**: the share of calls that returned an answer at all (not `error`).",
            "- **Invalid answer rate**: answers outside the allowed options, among calls that "
            "returned something. Counted as wrong, never corrected.",
            "- **Brier score**: mean of (probability − outcome)², with outcome 1 or 0. Measures "
            "whether stated probabilities can be trusted.",
            "- **Unwarranted disruptive action**: the engine chose restart or escalate when the "
            "expected action was observe or investigate.",
            "- **Missed escalation**: escalation was expected and the engine chose something else.",
            ""]
    return "\n".join(out) + "\n"
