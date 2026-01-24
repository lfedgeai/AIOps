#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import pandas as pd

HARNESS_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(os.getenv("AIOPS_OUT_DIR", str(HARNESS_DIR / "out")))
FEATURES_CSV = Path(os.getenv("OTEL_FEATURES_CSV", str(BASE_DIR / "features.csv")))
REPORT_JSON = Path(os.getenv("OTEL_REPORT_JSON", str(BASE_DIR / "report_isoforest.json")))
HTML_REPORT = Path(os.getenv("OTEL_OUT_HTML", str(BASE_DIR / "report_isoforest.html")))


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    with open(REPORT_JSON) as f:
        report = json.load(f)
    return df, report


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
        cell_w = w // 3
        cell_h = h // 3
        x0, y0 = cell_w, cell_h
        parts = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
        parts.append(f'<text x="{x0+cell_w/2:.1f}" y="{cell_h*0.6:.1f}" text-anchor="middle" font-size="12">pred: normal</text>')
        parts.append(f'<text x="{x0+cell_w*1.5:.1f}" y="{cell_h*0.6:.1f}" text-anchor="middle" font-size="12">pred: anomaly</text>')
        parts.append(f'<text x="{cell_w*0.5:.1f}" y="{y0+cell_h*0.8:.1f}" text-anchor="middle" font-size="12" transform="rotate(-90 {cell_w*0.5:.1f},{y0+cell_h*0.8:.1f})">true: normal</text>')
        parts.append(f'<text x="{cell_w*0.5:.1f}" y="{y0+cell_h*1.8:.1f}" text-anchor="middle" font-size="12" transform="rotate(-90 {cell_w*0.5:.1f},{y0+cell_h*1.8:.1f})">true: anomaly</text>')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="#f3f4f6" stroke="#d1d5db"/>')
        parts.append(f'<rect x="{x0+cell_w}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="#fef2f2" stroke="#d1d5db"/>')
        parts.append(f'<rect x="{x0}" y="{y0+cell_h}" width="{cell_w}" height="{cell_h}" fill="#fef2f2" stroke="#d1d5db"/>')
        parts.append(f'<rect x="{x0+cell_w}" y="{y0+cell_h}" width="{cell_w}" height="{cell_h}" fill="#ecfdf5" stroke="#d1d5db"/>')
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
            try:
                (BASE_DIR / "pr_curve.svg").write_text(_svg_polyline(xs, ys, stroke="#7c3aed"))
            except Exception:
                pass
        if roc.get("fpr") and roc.get("tpr"):
            xs = [max(0.0, min(1.0, float(x))) for x in roc["fpr"]]
            ys = [max(0.0, min(1.0, float(y))) for y in roc["tpr"]]
            roc_svg = _svg_polyline(xs, ys, stroke="#16a34a")
            try:
                (BASE_DIR / "roc_curve.svg").write_text(_svg_polyline(xs, ys, stroke="#16a34a"))
            except Exception:
                pass
    try:
        (BASE_DIR / "confusion_matrix.svg").write_text(_svg_confusion(tn, fp, fn, tp))
    except Exception:
        pass

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>IsolationForest Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 20px; }}
      .grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
      .kv {{ margin-bottom: 6px; }}
      table {{ border-collapse: collapse; }}
      th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; }}
    </style>
  </head>
  <body>
    <h1>IsolationForest Report</h1>
    <div class="kv">Train samples: <b>{num_train}</b></div>
    <div class="kv">Eval samples: <b>{num_eval}</b></div>
    <div class="kv">Precision: <b>{precision:.4f}</b> &nbsp;&nbsp; Recall: <b>{recall:.4f}</b> &nbsp;&nbsp; F1: <b>{f1:.4f}</b></div>
    <div class="kv">ROC-AUC: <b>{(roc_auc if roc_auc is not None else 'n/a')}</b> &nbsp;&nbsp; PR-AUC: <b>{(pr_auc if pr_auc is not None else 'n/a')}</b></div>
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
  </body>
</html>"""
    HTML_REPORT.parent.mkdir(parents=True, exist_ok=True)
    HTML_REPORT.write_text(html)


def main() -> int:
    _, report = load_data()
    write_html(report)
    print("Wrote:")
    print(f" - {HTML_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


