from __future__ import annotations

import html
import json
from pathlib import Path

from .metrics import EvaluationResult


def write_evaluation_report(result: EvaluationResult, out_dir: str | Path, title: str = "PSL model report") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "metrics.json"
    confusion_path = out / "confusion_matrix.svg"
    bars_path = out / "per_label_accuracy.svg"
    html_path = out / "index.html"

    metrics_payload = {
        "accuracy": result.accuracy,
        "sample_count": result.sample_count,
        "labels": result.labels,
        "confusion": result.confusion,
        "per_label_accuracy": result.per_label_accuracy,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    confusion_path.write_text(_confusion_svg(result), encoding="utf-8")
    bars_path.write_text(_bar_svg(result), encoding="utf-8")
    html_path.write_text(_html_report(title, result), encoding="utf-8")

    return {
        "metrics": str(metrics_path),
        "confusion_matrix": str(confusion_path),
        "per_label_accuracy": str(bars_path),
        "html": str(html_path),
    }


def _confusion_svg(result: EvaluationResult) -> str:
    labels = result.labels
    n = len(labels)
    cell = 42
    left = 140
    top = 90
    width = left + n * cell + 40
    height = top + n * cell + 120
    max_value = max((value for row in result.confusion for value in row), default=1) or 1

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        '<text x="24" y="34" fill="#e2e8f0" font-family="Arial" font-size="22" font-weight="700">Confusion Matrix</text>',
        '<text x="24" y="58" fill="#94a3b8" font-family="Arial" font-size="13">Rows: true labels, columns: predicted labels</text>',
    ]

    for i, label in enumerate(labels):
        safe = html.escape(label)
        y = top + i * cell + cell * 0.62
        x = left + i * cell + cell * 0.5
        parts.append(f'<text x="{left - 12}" y="{y:.1f}" fill="#cbd5e1" font-family="Arial" font-size="12" text-anchor="end">{safe}</text>')
        parts.append(f'<text x="{x:.1f}" y="{top - 12}" fill="#cbd5e1" font-family="Arial" font-size="12" text-anchor="start" transform="rotate(-55 {x:.1f} {top - 12})">{safe}</text>')

    for row_i, row in enumerate(result.confusion):
        for col_i, value in enumerate(row):
            intensity = value / max_value
            blue = int(50 + 180 * intensity)
            fill = f"rgb(30,{80 + blue // 2},{blue})"
            x = left + col_i * cell
            y = top + row_i * cell
            text_color = "#f8fafc" if intensity > 0.45 else "#cbd5e1"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" rx="6" fill="{fill}"/>')
            parts.append(f'<text x="{x + cell / 2:.1f}" y="{y + cell * 0.62:.1f}" fill="{text_color}" font-family="Arial" font-size="13" font-weight="700" text-anchor="middle">{value}</text>')

    parts.append(f'<text x="{left + n * cell / 2:.1f}" y="{height - 34}" fill="#94a3b8" font-family="Arial" font-size="13" text-anchor="middle">Accuracy: {result.accuracy:.2%} on {result.sample_count} samples</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _bar_svg(result: EvaluationResult) -> str:
    labels = result.labels
    row_h = 30
    width = 900
    left = 170
    top = 54
    height = top + len(labels) * row_h + 44
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#111827"/>',
        '<text x="24" y="34" fill="#e5e7eb" font-family="Arial" font-size="22" font-weight="700">Per-label Accuracy</text>',
    ]
    bar_w = width - left - 90
    for index, label in enumerate(labels):
        y = top + index * row_h
        acc = result.per_label_accuracy.get(label, 0.0)
        parts.append(f'<text x="{left - 12}" y="{y + 19}" fill="#d1d5db" font-family="Arial" font-size="13" text-anchor="end">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 6}" width="{bar_w}" height="16" rx="8" fill="#1f2937"/>')
        parts.append(f'<rect x="{left}" y="{y + 6}" width="{bar_w * acc:.1f}" height="16" rx="8" fill="#22c55e"/>')
        parts.append(f'<text x="{left + bar_w + 12}" y="{y + 19}" fill="#d1d5db" font-family="Arial" font-size="12">{acc:.0%}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _html_report(title: str, result: EvaluationResult) -> str:
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #020617; color: #e5e7eb; }}
    main {{ max-width: 1100px; margin: 32px auto; padding: 0 18px 48px; }}
    .card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 18px; padding: 22px; margin: 18px 0; }}
    .metric {{ font-size: 42px; font-weight: 800; color: #22c55e; }}
    img {{ max-width: 100%; border-radius: 14px; border: 1px solid #1e293b; }}
    code {{ color: #93c5fd; }}
  </style>
</head>
<body>
<main>
  <h1>{safe_title}</h1>
  <section class="card">
    <div>Top-1 accuracy</div>
    <div class="metric">{result.accuracy:.2%}</div>
    <p>{result.sample_count} evaluated samples, {len(result.labels)} labels.</p>
  </section>
  <section class="card">
    <h2>Confusion matrix</h2>
    <img src="confusion_matrix.svg" alt="Confusion matrix" />
  </section>
  <section class="card">
    <h2>Per-label accuracy</h2>
    <img src="per_label_accuracy.svg" alt="Per-label accuracy" />
  </section>
  <section class="card">
    <h2>Raw metrics</h2>
    <p>See <code>metrics.json</code>. Small, boring, easy to diff. My preferred kind of report.</p>
  </section>
</main>
</body>
</html>
"""
