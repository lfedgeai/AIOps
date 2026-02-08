#!/usr/bin/env python3
from __future__ import annotations
import json
import glob
from pathlib import Path
from typing import List, Tuple, Dict

BASE_DIR = Path("/home/redhat/git/OTEL/otel-demo/otel_ground_truth_data")


def _hbar_svg(items: List[Tuple[str, float, float]], width: int = 720, bar_h: int = 22, pad: int = 16):
    """
    items: list of (feature, mean, std). Renders a horizontal bar chart with error whiskers.
    """
    if not items:
        return "<svg/>"
    max_val = max((v for _, v, _ in items), default=1.0) or 1.0
    h = pad * 2 + bar_h * len(items)
    w = width
    def sx(v: float) -> float:
        return pad + (v / max_val) * (w - 2 * pad)
    lines = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    # axis
    lines.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#999" stroke-width="1"/>')
    # bars
    for i, (name, mean, std) in enumerate(items):
        y = pad + i * bar_h + 4
        x0 = pad
        x1 = sx(mean)
        lines.append(f'<rect x="{x0}" y="{y}" width="{max(1, x1-x0):.1f}" height="{bar_h-8}" fill="#2563eb" opacity="0.85"/>')
        # error whisker (std)
        if std and std > 0:
            xe = sx(min(max_val, mean + std))
            lines.append(f'<line x1="{x1}" y1="{y+ (bar_h-8)/2:.1f}" x2="{xe:.1f}" y2="{y+ (bar_h-8)/2:.1f}" stroke="#1d4ed8" stroke-width="2"/>')
        # labels
        label = f'{name} ({mean:.3f})'
        lines.append(f'<text x="{pad+4}" y="{y+bar_h-10}" font-size="12" fill="#111">{label}</text>')
    lines.append('</svg>')
    return "\n".join(lines)


def main() -> int:
    reports = sorted(glob.glob(str(BASE_DIR / "report_isoforest_perm_*.json")))
    if not reports:
        print("No permutation reports found.")
        return 1
    perm_path = Path(reports[-1])
    data = json.loads(Path(perm_path).read_text())
    ranked = data.get("importances_perm_ranked", [])
    # ranked is list of [feature, {mean_drop_pr_auc, std_drop_pr_auc, n}]
    rows: List[Tuple[str, float, float]] = []
    for item in ranked:
        if not isinstance(item, list) or len(item) < 2:
            continue
        name = str(item[0])
        stats: Dict[str, float] = item[1] or {}
        rows.append((name, float(stats.get("mean_drop_pr_auc", 0.0)), float(stats.get("std_drop_pr_auc", 0.0))))
    # HTML
    title = "IsolationForest Permutation Importances"
    hbars = _hbar_svg(rows)
    table_rows = "\n".join(
        f"<tr><td>{i+1}</td><td>{n}</td><td>{m:.6f}</td><td>{s:.6f}</td></tr>"
        for i, (n, m, s) in enumerate(rows)
    )
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111; }}
h1 {{ margin: 0 0 8px 0; font-size: 20px; }}
.sub {{ color: #555; margin-bottom: 16px; }}
table {{ border-collapse: collapse; font-size: 14px; }}
td, th {{ border: 1px solid #ddd; padding: 6px 8px; }}
th {{ background: #f5f5f5; }}
</style>
<h1>{title}</h1>
<div class="sub">
Model: {data.get("model_path","")}<br/>
Baseline PR-AUC: {data.get("baseline_pr_auc")}<br/>
Repeats: {data.get("perm_repeats")} | Eval: {data.get("eval_count")} | Train: {data.get("train_count")}
</div>
<h2>Importance (mean ΔPR-AUC)</h2>
{hbars}
<h2>Details</h2>
<table>
  <thead><tr><th>#</th><th>Feature</th><th>mean_drop_pr_auc</th><th>std_drop_pr_auc</th></tr></thead>
  <tbody>
  {table_rows}
  </tbody>
</table>
"""
    out_html = perm_path.with_suffix(".html")
    out_html.write_text(html)
    print(f"Wrote: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


