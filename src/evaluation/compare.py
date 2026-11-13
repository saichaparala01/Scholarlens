"""Collect every results/*.json file into comparison tables for the report.

    python -m src.evaluation.compare

Writes results/comparison.md (paste into the report), results/comparison.csv,
and one bar chart per task in docs/ (comparison_<task>.png).
"""

import csv
import json
from pathlib import Path

from src.data_loading import load_config
from src.utils import ROOT

TASK_TITLES = {"summarization": "Summarization (arXiv)", "qa": "Question Answering (SQuAD 2.0)",
               "ner": "Named Entity Recognition"}


def load_results(results_dir: Path) -> dict:
    """Group the metric files by task: {task: [payload, ...]}."""
    grouped = {}
    for path in sorted(results_dir.glob("*__*.json")):
        task = path.name.split("__")[0]
        with open(path) as f:
            grouped.setdefault(task, []).append(json.load(f))
    return grouped


def markdown_table(rows: list) -> str:
    metric_names = sorted({k for r in rows for k in r["metrics"]})
    header = "| Model | " + " | ".join(metric_names) + " | Train time (min) |"
    sep = "|" + "---|" * (len(metric_names) + 2)
    lines = [header, sep]
    for r in rows:
        values = [str(r["metrics"].get(m, "-")) for m in metric_names]
        lines.append(f"| {r['model']} | " + " | ".join(values) + f" | {r.get('train_minutes', '-')} |")
    return "\n".join(lines)


def make_chart(task: str, rows: list, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metric_names = [m for m in sorted({k for r in rows for k in r["metrics"]})
                    if all(isinstance(r["metrics"].get(m), (int, float)) for r in rows)]
    if not metric_names:
        return
    width = 0.8 / len(rows)
    fig, ax = plt.subplots(figsize=(1.6 * len(metric_names) + 3, 4))
    for i, r in enumerate(rows):
        xs = [j + i * width for j in range(len(metric_names))]
        ax.bar(xs, [r["metrics"][m] for m in metric_names], width, label=r["model"])
    ax.set_xticks([j + width * (len(rows) - 1) / 2 for j in range(len(metric_names))])
    ax.set_xticklabels(metric_names, rotation=20)
    ax.set_ylabel("Score")
    ax.set_title(TASK_TITLES.get(task, task))
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    results_dir = ROOT / cfg["paths"]["results_dir"]
    grouped = load_results(results_dir)
    if not grouped:
        print(f"No result files found in {results_dir}. Train a model first.")
        return

    md, csv_rows = [], []
    for task in ("summarization", "qa", "ner"):
        rows = grouped.get(task)
        if not rows:
            md.append(f"## {TASK_TITLES[task]}\n\n_No results yet._\n")
            continue
        md.append(f"## {TASK_TITLES[task]}\n\n{markdown_table(rows)}\n")
        make_chart(task, rows, ROOT / "docs" / f"comparison_{task}.png")
        for r in rows:
            for metric, value in r["metrics"].items():
                csv_rows.append([task, r["model"], metric, value])

    (results_dir / "comparison.md").write_text("\n".join(md))
    with open(results_dir / "comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "model", "metric", "value"])
        writer.writerows(csv_rows)
    print("\n".join(md))
    print(f"Saved {results_dir / 'comparison.md'} and comparison.csv")


if __name__ == "__main__":
    main()
