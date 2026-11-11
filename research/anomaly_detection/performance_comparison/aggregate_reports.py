#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def load_runs(out_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for d in sorted(out_dir.glob("ad_compare_*")):
        if not d.is_dir():
            continue
        j = d / "ad_compare.json"
        if not j.exists():
            continue
        try:
            data = json.loads(j.read_text())
        except Exception:
            continue
        ts = d.name.replace("ad_compare_", "")
        # Try to parse readable time
        dt = None
        try:
            dt = datetime.strptime(ts, "%Y%m%d%H%M%S")
        except Exception:
            pass
        runs.append(
            {
                "dir": d.name,
                "path": str(d),
                "timestamp": ts,
                "datetime": (dt.isoformat() if dt else None),
                "data": data,
            }
        )
    # Latest first
    runs.sort(key=lambda r: r["timestamp"], reverse=True)
    return runs


def write_json(out_dir: Path, runs: List[Dict[str, Any]]) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "num_runs": len(runs),
        "runs": [
            {
                "dir": r["dir"],
                "timestamp": r["timestamp"],
                "datetime": r["datetime"],
                "ad_compare_json": str(Path(r["path"]) / "ad_compare.json"),
                "ad_compare_html": str(Path(r["path"]) / "ad_compare.html"),
                "dataset_validation": r["data"].get("dataset_validation"),
                "results": r["data"].get("results"),
            }
            for r in runs
        ],
    }
    (out_dir / "ad_compare_rolling.json").write_text(json.dumps(payload, indent=2))


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def write_html(out_dir: Path, runs: List[Dict[str, Any]]) -> None:
    def _svg_polyline(xs, ys, w: int = 360, h: int = 240, pad: int = 20, stroke: str = "#2563eb") -> str:
        def sx(x: float) -> float:
            return pad + float(x) * (w - 2 * pad)

        def sy(y: float) -> float:
            return h - pad - float(y) * (h - 2 * pad)

        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        axes = (
            f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#999" stroke-width="1"/>'
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#999" stroke-width="1"/>'
        )
        poly = f'<polyline fill="none" stroke="{stroke}" stroke-width="2" points="{pts}"/>'
        return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{axes}{poly}</svg>'

    def _quality_label(v: float) -> str:
        if v >= 0.80:
            return "strong"
        if v >= 0.65:
            return "moderate"
        if v >= 0.50:
            return "weak"
        return "poor"

    def _commentary(det: Dict[str, Any]) -> str:
        p = det.get("precision")
        r = det.get("recall")
        f1 = det.get("f1")
        pr_auc = det.get("pr_auc")
        roc_auc = det.get("roc_auc")
        parts = []
        if pr_auc is not None:
            parts.append(f"PR-AUC {pr_auc:.3f} ({_quality_label(float(pr_auc))} ranking of positives).")
        else:
            parts.append("PR-AUC n/a (insufficient positives).")
        if roc_auc is not None:
            parts.append(f"ROC-AUC {roc_auc:.3f} ({_quality_label(float(roc_auc))} separability).")
        else:
            parts.append("ROC-AUC n/a.")
        # Operating point hint
        if isinstance(p, (int, float)) and isinstance(r, (int, float)):
            if p >= 0.95 and r <= 0.4:
                parts.append("Operating point is conservative (high precision, low recall) — likely due to a high threshold quantile.")
            elif r >= 0.7 and p <= 0.6:
                parts.append("Operating point favors recall over precision — consider raising the threshold to reduce false positives.")
        # Generic shape guidance
        parts.append("PR curve bows toward top-right when many true faults rank above baselines; a flatter line indicates noisy ranking or class imbalance.")
        parts.append("ROC curve bows toward top-left when scores separate classes well; a diagonal (~0.5) suggests near-random separation.")
        return " ".join(parts)

    rows = []
    sections = []
    metrics_html = """
  <div style="margin:10px 0 16px 0;padding:10px;border:1px solid #e5e7eb;border-radius:6px;">
    <h2 style="margin:0 0 8px 0;font-size:16px;">Metric definitions</h2>
    <ul style="margin:0 0 0 18px;">
      <li><b>Precision</b>: TP / (TP + FP) — of predicted anomalies, how many are true faults.</li>
      <li><b>Recall</b>: TP / (TP + FN) — of true faults, how many were detected.</li>
      <li><b>F1</b>: harmonic mean of precision and recall — balances both.</li>
      <li><b>PR-AUC</b>: area under Precision–Recall curve — ranking quality for the positive class.</li>
      <li><b>ROC-AUC</b>: area under ROC curve — overall class separability.</li>
    </ul>
  </div>"""
    # Build prev metrics map per run/ detector for deltas (use chronological order)
    prev_map: Dict[str, Dict[str, Dict[str, float]]] = {}
    last_by_detector: Dict[str, Dict[str, float]] = {}
    for r in reversed(runs):  # oldest → newest
        res = (r.get("data") or {}).get("results") or []
        prev_map[r["dir"]] = {}
        for det in res:
            name = str(det.get("detector", ""))
            prev_map[r["dir"]][name] = last_by_detector.get(name, {})
            try:
                last_by_detector[name] = {
                    "f1": float(det.get("f1")) if det.get("f1") is not None else None,  # type: ignore
                    "pr_auc": float(det.get("pr_auc")) if det.get("pr_auc") is not None else None,  # type: ignore
                }
            except Exception:
                pass

    # Build per-detector trends over time (chronological)
    trends: Dict[str, Dict[str, List[float]]] = {}
    for r in reversed(runs):
        res = (r.get("data") or {}).get("results") or []
        for det in res:
            name = str(det.get("detector", ""))
            trends.setdefault(name, {"f1": [], "pr_auc": []})
            try:
                trends[name]["f1"].append(float(det.get("f1")) if det.get("f1") is not None else None)  # type: ignore
            except Exception:
                trends[name]["f1"].append(None)  # type: ignore
            try:
                trends[name]["pr_auc"].append(float(det.get("pr_auc")) if det.get("pr_auc") is not None else None)  # type: ignore
            except Exception:
                trends[name]["pr_auc"].append(None)  # type: ignore

    # Render trends (simple sparkline-like charts)
    trend_cards = []
    for det_name, series in trends.items():
        f1_vals = [v for v in series["f1"] if isinstance(v, (float, int))]
        pr_vals = [v for v in series["pr_auc"] if isinstance(v, (float, int))]
        # Normalize x positions 0..1
        n = max(len(series["f1"]), 1)
        xs = [i / max(1, (n - 1)) for i in range(n)]
        def safe_poly(vals, stroke):
            ys = []
            xs_eff = []
            for x, v in zip(xs, series[vals]):  # type: ignore
                if isinstance(v, (float, int)):
                    xs_eff.append(x)
                    ys.append(float(v))
            if not xs_eff:
                return '<div style="color:#666">n/a</div>'
            # Clamp to [0,1]
            xs_c = [max(0.0, min(1.0, float(x))) for x in xs_eff]
            ys_c = [max(0.0, min(1.0, float(y))) for y in ys]
            return _svg_polyline(xs_c, ys_c, w=300, h=120, pad=18, stroke=stroke)
        trend_cards.append(f"""
    <div style="border:1px solid #e5e7eb;border-radius:8px;padding:10px;">
      <div style="font-weight:600;margin-bottom:6px">{_html_escape(det_name)}</div>
      <div style="display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:10px;">
        <div><div style="font-size:12px;color:#555;margin-bottom:2px">F1 over time (old→new)</div>{safe_poly("f1", "#0ea5e9")}</div>
        <div><div style="font-size:12px;color:#555;margin-bottom:2px">PR-AUC over time (old→new)</div>{safe_poly("pr_auc", "#16a34a")}</div>
      </div>
    </div>
""")
    for idx, r in enumerate(runs, start=1):
        d = r["data"]
        val = d.get("dataset_validation") or {}
        q = d.get("threshold_quantile")
        mv = (val.get("missing_files") or {}) if isinstance(val, dict) else {}
        ev = (val.get("empties") or {}) if isinstance(val, dict) else {}
        results = d.get("results") or []
        # Summary row (first detector shown)
        first = results[0] if results else {}
        rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td><a href=\"{_html_escape(r['dir'])}/ad_compare.html\">{_html_escape(r['dir'])}</a></td>"
            f"<td>{_html_escape(str(r.get('datetime') or r['timestamp']))}</td>"
            f"<td>{_html_escape('n/a' if q is None else f'{float(q):.4f}')}</td>"
            f"<td>{_html_escape(str(val.get('total_samples','')))}</td>"
            f"<td>{_html_escape(str(val.get('num_train','')))}</td>"
            f"<td>{_html_escape(str(val.get('num_eval','')))}</td>"
            f"<td>{_html_escape(str(first.get('detector','')))}</td>"
            f"<td>{first.get('precision','')}</td>"
            f"<td>{first.get('recall','')}</td>"
            f"<td>{first.get('f1','')}</td>"
            "</tr>"
        )
        # Detailed section per run
        det_rows = []
        det_graphs = []
        # Combined vs standalone table (per run)
        comp_html = ""
        try:
            fams = {
                "IsolationForest": {"combined": "IsolationForest(sklearn)", "logs": "IsolationForest(logs)", "traces": "IsolationForest(traces)", "metrics": "IsolationForest(metrics)"},
                "COPOD": {"combined": "COPOD(pyod)", "logs": "COPOD(logs)", "traces": "COPOD(traces)", "metrics": "COPOD(metrics)"},
                "RRCF": {"combined": "RRCF(rrcf)", "logs": "RRCF(logs)", "traces": "RRCF(traces)", "metrics": "RRCF(metrics)"},
            }
            f1map = {str(det.get("detector","")): det.get("f1") for det in results}
            head = "<tr><th>Algorithm</th><th>Combined F1</th><th>Logs F1</th><th>Traces F1</th><th>Metrics F1</th></tr>"
            rows_cmp = []
            for fam, names in fams.items():
                def fmt(name: str) -> str:
                    v = f1map.get(name, None)
                    return "" if v is None else f"{float(v):.4f}"
                rows_cmp.append(
                    "<tr>"
                    f"<td>{fam}</td>"
                    f"<td>{fmt(names['combined'])}</td>"
                    f"<td>{fmt(names['logs'])}</td>"
                    f"<td>{fmt(names['traces'])}</td>"
                    f"<td>{fmt(names['metrics'])}</td>"
                    "</tr>"
                )
            comp_html = f"""
    <div style="margin:8px 0 10px 0;padding:10px;border:1px solid #e5e7eb;border-radius:6px;">
      <div style="font-weight:600;margin-bottom:6px">Combined vs standalone (F1)</div>
      <table style="border-collapse:collapse;margin-top:4px">
        {head}
        {"".join(rows_cmp)}
      </table>
    </div>
    """
        except Exception:
            comp_html = ""
        for det in results:
            # Compute deltas vs previous run for this detector
            prev = (prev_map.get(r["dir"], {}) or {}).get(str(det.get("detector","")), {})
            def fmt_with_delta(val, prev_val):
                try:
                    if prev_val is None or val is None:
                        return f"{'' if val is None else f'{val:.4f}'}"
                    delta = float(val) - float(prev_val)
                    color = "#16a34a" if delta > 1e-9 else ("#ef4444" if delta < -1e-9 else "#6b7280")
                    sign = "+" if delta > 0 else ""
                    return f"{val:.4f} <span style='color:{color};font-size:12px'>({sign}{delta:.4f})</span>"
                except Exception:
                    return f"{'' if val is None else f'{val:.4f}'}"
            det_rows.append(
                "<tr>"
                f"<td>{_html_escape(str(det.get('detector','')))}</td>"
                f"<td>{'' if det.get('precision') is None else f'{det.get('precision'):.4f}'}</td>"
                f"<td>{'' if det.get('recall') is None else f'{det.get('recall'):.4f}'}</td>"
                f"<td>{fmt_with_delta(det.get('f1'), (prev or {}).get('f1'))}</td>"
                f"<td>{'n/a' if det.get('pr_auc') is None else fmt_with_delta(det.get('pr_auc'), (prev or {}).get('pr_auc'))}</td>"
                f"<td>{'n/a' if det.get('roc_auc') is None else f'{det.get('roc_auc'):.4f}'}</td>"
                f"<td>{det.get('threshold','')}</td>"
                "</tr>"
            )
            # Curves
            pr_svg = ""
            roc_svg = ""
            try:
                pr = det.get("pr_curve") or {}
                if pr and pr.get("recall") and pr.get("precision"):
                    xs = [max(0.0, min(1.0, float(x))) for x in pr.get("recall", [])]
                    ys = [max(0.0, min(1.0, float(y))) for y in pr.get("precision", [])]
                    pr_svg = _svg_polyline(xs, ys, stroke="#7c3aed")
            except Exception:
                pass
            try:
                roc = det.get("roc_curve") or {}
                if roc and roc.get("fpr") and roc.get("tpr"):
                    xs = [max(0.0, min(1.0, float(x))) for x in roc.get("fpr", [])]
                    ys = [max(0.0, min(1.0, float(y))) for y in roc.get("tpr", [])]
                    roc_svg = _svg_polyline(xs, ys, stroke="#16a34a")
            except Exception:
                pass
            comment = _commentary(det)
            # Per-flag table if available
            pf_rows_html = ""
            try:
                pf = det.get("by_flag") or []
                if pf:
                    rows_pf = []
                    for it in pf:
                        try:
                            rows_pf.append(
                                f"<tr><td>{_html_escape(str(it.get('flag','')))}</td>"
                                f"<td>{_html_escape(str(it.get('support_pos','')))}</td>"
                                f"<td>{float(it.get('precision',0)):.4f}</td>"
                                f"<td>{float(it.get('recall',0)):.4f}</td>"
                                f"<td>{float(it.get('f1',0)):.4f}</td></tr>"
                            )
                        except Exception:
                            continue
                    if rows_pf:
                        pf_rows_html = f"""
        <div style="margin-top:6px">Per-flag metrics (positives only):</div>
        <table style="border-collapse:collapse;margin-top:4px">
          <tr><th style="border:1px solid #ccc;padding:4px 8px;">flag</th>
              <th style="border:1px solid #ccc;padding:4px 8px;">positive windows</th>
              <th style="border:1px solid #ccc;padding:4px 8px;">P</th>
              <th style="border:1px solid #ccc;padding:4px 8px;">R</th>
              <th style="border:1px solid #ccc;padding:4px 8px;">F1</th></tr>
          {"".join(rows_pf)}
        </table>"""
            except Exception:
                pass
            det_graphs.append(
                f"""
      <div style="border:1px solid #e5e7eb;border-radius:6px;padding:8px;">
        <div style="font-weight:600;margin-bottom:4px">{_html_escape(str(det.get('detector','')))}</div>
        <div style="display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:10px;">
          <div><div style="font-size:12px;color:#555;margin-bottom:2px">Precision–Recall</div>{(pr_svg or '<div style="color:#666">n/a</div>')}</div>
          <div><div style="font-size:12px;color:#555;margin-bottom:2px">ROC</div>{(roc_svg or '<div style="color:#666">n/a</div>')}</div>
        </div>
        <div style="margin-top:6px;font-size:12px;color:#374151;">{_html_escape(comment)}</div>
        {pf_rows_html}
      </div>
"""
            )
        sections.append(
            f"""
  <div style="margin:16px 0 24px 0;padding:12px;border:1px solid #e5e7eb;border-radius:8px;">
    <h2 style="margin:0 0 6px 0;font-size:18px;">{_html_escape(r['dir'])}</h2>
    <div style="color:#555;margin-bottom:8px">{_html_escape(str(r.get('datetime') or r['timestamp']))}</div>
    <div style="display:grid;grid-template-columns:repeat(3,minmax(160px,1fr));gap:8px;margin-bottom:8px;">
      <div>Total samples: <b>{_html_escape(str(val.get('total_samples','n/a')))}</b></div>
      <div>Train rows: <b>{_html_escape(str(val.get('num_train','n/a')))}</b></div>
      <div>Eval rows: <b>{_html_escape(str(val.get('num_eval','n/a')))}</b></div>
      <div>Missing logs: <b>{_html_escape(str(mv.get('logs','n/a')))}</b></div>
      <div>Missing traces: <b>{_html_escape(str(mv.get('traces','n/a')))}</b></div>
      <div>Missing metrics: <b>{_html_escape(str(mv.get('metrics','n/a')))}</b></div>
      <div>Zero-line logs: <b>{_html_escape(str(ev.get('logs_zero_lines','n/a')))}</b></div>
      <div>Zero-count traces: <b>{_html_escape(str(ev.get('traces_zero_count','n/a')))}</b></div>
      <div></div>
    </div>
    {comp_html}
    <div>Per-detector metrics:</div>
    <table style="border-collapse:collapse;margin-top:6px">
      <tr><th style="border:1px solid #ccc;padding:6px 10px;">detector</th>
          <th style="border:1px solid #ccc;padding:6px 10px;">precision</th>
          <th style="border:1px solid #ccc;padding:6px 10px;">recall</th>
          <th style="border:1px solid #ccc;padding:6px 10px;">f1</th>
          <th style="border:1px solid #ccc;padding:6px 10px;">PR-AUC</th>
          <th style="border:1px solid #ccc;padding:6px 10px;">ROC-AUC</th>
          <th style="border:1px solid #ccc;padding:6px 10px;">threshold</th></tr>
      {"".join(det_rows)}
    </table>
    <div style="margin-top:8px;display:flex;flex-direction:column;gap:14px;">
      {"".join(det_graphs)}
    </div>
    <div style="margin-top:6px">
      <a href="{_html_escape(r['dir'])}/ad_compare.html">Open full run report</a>
    </div>
  </div>
"""
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Rolling Anomaly Detector Comparison</title>
<style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px;text-align:center}}</style>
</head><body>
  <h1>Rolling Anomaly Detector Comparison</h1>
  <div style="margin:8px 0 12px 0;color:#555">Latest runs are shown first. Each run links to its full report.</div>
  {metrics_html}
  <table>
    <tr><th>#</th><th>run</th><th>time</th><th>quantile</th><th>samples</th><th>train</th><th>eval</th><th>first detector</th><th>P</th><th>R</th><th>F1</th></tr>
    {"".join(rows)}
  </table>
  {"".join(sections)}
</body></html>"""
    (out_dir / "ad_compare_rolling.html").write_text(html)


def main() -> int:
    root = Path(__file__).resolve().parent
    out_dir = root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(out_dir)
    write_json(out_dir, runs)
    write_html(out_dir, runs)
    print(f"Updated rolling report with {len(runs)} runs:")
    print(f" - {out_dir/'ad_compare_rolling.json'}")
    print(f" - {out_dir/'ad_compare_rolling.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

