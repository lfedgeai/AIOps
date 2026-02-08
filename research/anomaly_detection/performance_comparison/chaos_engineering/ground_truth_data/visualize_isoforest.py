#!/usr/bin/env python3
"""
Visualize IsolationForest report and features (HTML-only, no external libs).
Creates:
- report_isoforest.html (summary page with CSS-based bars and tables)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/home/redhat/git/OTEL/otel-demo/otel_ground_truth_data")
# Allow overriding inputs/outputs via environment variables
FEATURES_CSV = Path(os.getenv("OTEL_FEATURES_CSV", str(BASE_DIR / "features.csv")))
REPORT_JSON = Path(os.getenv("OTEL_REPORT_JSON", str(BASE_DIR / "report_isoforest.json")))
HTML_REPORT = Path(os.getenv("OTEL_OUT_HTML", str(BASE_DIR / "report_isoforest.html")))


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    with open(REPORT_JSON) as f:
        report = json.load(f)
    return df, report


def compute_eval_predictions(df: pd.DataFrame) -> pd.DataFrame:
    # Recompute predictions to get per-sample scores for visualization
    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        # If sklearn is not present, just mark anomalies by root_cause != none
        eval_df = df[df["is_baseline"] == False].copy()  # noqa: E712
        eval_df["pred"] = (eval_df["root_cause"] != "none").astype(int)
        eval_df["anomaly_score"] = 0.0
        return eval_df

    feature_cols = ["trace_count", "log_error_count", "metrics_series_len"]
    train_df = df[df["is_baseline"] == True].copy()  # noqa: E712
    eval_df = df[df["is_baseline"] == False].copy()  # noqa: E712
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    model.fit(train_df[feature_cols])
    eval_df = eval_df.copy()
    eval_df["pred"] = (model.predict(eval_df[feature_cols]) == -1).astype(int)
    eval_df["anomaly_score"] = -model.decision_function(eval_df[feature_cols])
    return eval_df


def write_html(report: dict):
    precision = report.get("precision")
    recall = report.get("recall")
    f1 = report.get("f1")
    num_train = report.get("num_train")
    num_eval = report.get("num_eval")
    cm = report.get("confusion_matrix", {})
    tn, fp, fn, tp = cm.get("tn", 0), cm.get("fp", 0), cm.get("fn", 0), cm.get("tp", 0)
    roc_auc = report.get("roc_auc")
    pr_auc = report.get("pr_auc")
    curves = report.get("curves", {})

    def _svg_polyline(xs, ys, w=360, h=240, pad=20, stroke="#2563eb"):
        def sx(x): return pad + x * (w - 2*pad)
        def sy(y): return h - pad - y * (h - 2*pad)
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        axes = f'''
          <line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#999" stroke-width="1"/>
          <line x1="{pad}" y1="{pad}"   x2="{pad}"   y2="{h-pad}" stroke="#999" stroke-width="1"/>
        '''
        poly = f'<polyline fill="none" stroke="{stroke}" stroke-width="2" points="{pts}"/>'
        return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{axes}{poly}</svg>'

    def _svg_confusion(tn: int, fp: int, fn: int, tp: int, w: int = 360, h: int = 180):
        # Simple 2x2 grid with counts
        cell_w = w // 3
        cell_h = h // 3
        x0, y0 = cell_w, cell_h
        # headers + cells
        parts = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
        # headers
        parts.append(f'<text x="{x0+cell_w/2:.1f}" y="{cell_h*0.6:.1f}" text-anchor="middle" font-size="12">pred: normal</text>')
        parts.append(f'<text x="{x0+cell_w*1.5:.1f}" y="{cell_h*0.6:.1f}" text-anchor="middle" font-size="12">pred: anomaly</text>')
        parts.append(f'<text x="{cell_w*0.5:.1f}" y="{y0+cell_h*0.8:.1f}" text-anchor="middle" font-size="12" transform="rotate(-90 {cell_w*0.5:.1f},{y0+cell_h*0.8:.1f})">true: normal</text>')
        parts.append(f'<text x="{cell_w*0.5:.1f}" y="{y0+cell_h*1.8:.1f}" text-anchor="middle" font-size="12" transform="rotate(-90 {cell_w*0.5:.1f},{y0+cell_h*1.8:.1f})">true: anomaly</text>')
        # grid rectangles
        parts.append(f'<rect x="{x0}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="#f3f4f6" stroke="#d1d5db"/>')  # TN
        parts.append(f'<rect x="{x0+cell_w}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="#fef2f2" stroke="#d1d5db"/>')  # FP
        parts.append(f'<rect x="{x0}" y="{y0+cell_h}" width="{cell_w}" height="{cell_h}" fill="#fef2f2" stroke="#d1d5db"/>')  # FN
        parts.append(f'<rect x="{x0+cell_w}" y="{y0+cell_h}" width="{cell_w}" height="{cell_h}" fill="#ecfdf5" stroke="#d1d5db"/>')  # TP
        # counts
        parts.append(f'<text x="{x0+cell_w/2:.1f}" y="{y0+cell_h/2:.1f}" text-anchor="middle" font-size="14">{tn}</text>')
        parts.append(f'<text x="{x0+cell_w*1.5:.1f}" y="{y0+cell_h/2:.1f}" text-anchor="middle" font-size="14">{fp}</text>')
        parts.append(f'<text x="{x0+cell_w/2:.1f}" y="{y0+cell_h*1.5:.1f}" text-anchor="middle" font-size="14">{fn}</text>')
        parts.append(f'<text x="{x0+cell_w*1.5:.1f}" y="{y0+cell_h*1.5:.1f}" text-anchor="middle" font-size="14">{tp}</text>')
        parts.append('</svg>')
        return "".join(parts)

    pr_svg = ""
    roc_svg = ""
    if isinstance(curves, dict):
        pr = curves.get("pr", {})
        roc = curves.get("roc", {})
        if pr.get("precision") and pr.get("recall"):
            xs = [max(0.0, min(1.0, float(x))) for x in pr["recall"]]
            ys = [max(0.0, min(1.0, float(y))) for y in pr["precision"]]
            pr_svg = _svg_polyline(xs, ys, stroke="#7c3aed")
            # Also export standalone PR curve SVG
            try:
                (BASE_DIR / "pr_curve.svg").write_text(_svg_polyline(xs, ys, stroke="#7c3aed"))
            except Exception:
                pass
        if roc.get("fpr") and roc.get("tpr"):
            xs = [max(0.0, min(1.0, float(x))) for x in roc["fpr"]]
            ys = [max(0.0, min(1.0, float(y))) for y in roc["tpr"]]
            roc_svg = _svg_polyline(xs, ys, stroke="#16a34a")
            # Also export standalone ROC curve SVG
            try:
                (BASE_DIR / "roc_curve.svg").write_text(_svg_polyline(xs, ys, stroke="#16a34a"))
            except Exception:
                pass
    # Export confusion matrix SVG
    try:
        (BASE_DIR / "confusion_matrix.svg").write_text(_svg_confusion(tn, fp, fn, tp))
    except Exception:
        pass

    # Tree-based importances (optional)
    importances_tbl = ""
    fim = report.get("feature_importances_ranked") or []
    if isinstance(fim, list) and fim:
        rows = []
        for item in fim:
            # item can be ["feature", importance] or ["feature", value]
            try:
                name = str(item[0])
                val = float(item[1])
                rows.append(f"<tr><td>{name}</td><td>{val:.6f}</td></tr>")
            except Exception:
                continue
        if rows:
            importances_tbl = """
    <h3>Feature Importances (tree)</h3>
    <table>
      <tr><th>feature</th><th>importance</th></tr>
      {rows}
    </table>
    """.format(rows="\n      ".join(rows))

    # Inline styles and CSS-based bars for anomaly scores
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>IsolationForest Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 20px; }}
      .grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
      .kv {{ margin-bottom: 6px; }}
      .mono {{ font-family: monospace; }}
      table {{ border-collapse: collapse; }}
      th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; }}
      .bar {{ height: 14px; background: #e0e7ff; border: 1px solid #c7d2fe; }}
      .bar-inner {{ height: 100%; background: #7c3aed; }}
      .anomaly {{ color: #b91c1c; font-weight: bold; }}
      .normal {{ color: #1d4ed8; }}
    </style>
  </head>
  <body>
    <h1>IsolationForest Report</h1>
    <div class="kv">Train samples: <b>{num_train}</b></div>
    <div class="kv">Eval samples: <b>{num_eval}</b></div>
    <div class="kv">Precision: <b>{precision:.4f}</b> &nbsp;&nbsp; Recall: <b>{recall:.4f}</b> &nbsp;&nbsp; F1: <b>{f1:.4f}</b></div>
    <div class="kv">ROC-AUC: <b>{(roc_auc if roc_auc is not None else 'n/a')}</b> &nbsp;&nbsp; PR-AUC: <b>{(pr_auc if pr_auc is not None else 'n/a')}</b></div>
    <h3>Confusion Matrix</h3>
    <table>
      <tr>
        <th></th><th>pred: normal</th><th>pred: anomaly</th>
      </tr>
      <tr>
        <th>true: normal</th><td>{tn}</td><td>{fp}</td>
      </tr>
      <tr>
        <th>true: anomaly</th><td>{fn}</td><td>{tp}</td>
      </tr>
    </table>
    <div class="grid" style="margin-top:14px">
      <div>
        <h3>Precision–Recall Curve</h3>
        {(pr_svg or '<div style="color:#666">n/a</div>')}
      </div>
      <div>
        <h3>ROC Curve</h3>
        {(roc_svg or '<div style="color:#666">n/a</div>')}
      </div>
    </div>
    {importances_tbl or ""}
    <h3>Eval Anomaly Scores</h3>
    <div id="scores"></div>
    <script>
      // Build a lightweight bar chart with divs
      const scores = [];
    </script>
  </body>
</html>"""
    HTML_REPORT.write_text(html)


def main() -> int:
    df, report = load_data()
    eval_df = compute_eval_predictions(df)
    # Append scores table into HTML with simple inline script
    write_html(report)
    if not eval_df.empty:
        # Normalize scores to [0,1] to compute widths
        s = eval_df["anomaly_score"].astype(float)
        s_min, s_max = float(s.min()), float(s.max())
        def norm(x: float) -> float:
            if s_max <= s_min:
                return 0.5
            return (x - s_min) / (s_max - s_min)
        rows = []
        for _, r in eval_df.sort_values("anomaly_score", ascending=False).iterrows():
            label = str(r["label"])
            score = float(r["anomaly_score"])
            pred = int(r["pred"])
            width = int(20 + 80 * norm(score))  # width in percent
            cls = "anomaly" if pred == 1 else "normal"
            bar_html = f'<div class="bar"><div class="bar-inner" style="width:{width}%"></div></div>'
            rows.append(f'<tr><td class="{cls}">{label}</td><td>{score:.4f}</td><td>{bar_html}</td></tr>')
        table_html = """
<table>
  <tr><th>label</th><th>score</th><th>bar</th></tr>
  {rows}
</table>
""".format(rows="\n  ".join(rows))
        # Inject into the placeholder div
        html = HTML_REPORT.read_text()
        html = html.replace(
            '<div id="scores"></div>',
            f'<div id="scores">{table_html}</div>'
        )
        HTML_REPORT.write_text(html)
    print("Wrote:")
    print(f" - {HTML_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


