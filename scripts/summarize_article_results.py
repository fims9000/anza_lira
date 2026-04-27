#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = [
    ("test_dice", "Dice"),
    ("test_iou", "IoU"),
    ("test_cldice", "clDice"),
    ("test_precision", "Precision"),
    ("test_recall", "Recall"),
    ("test_specificity", "Specificity"),
    ("test_balanced_accuracy", "Balanced Acc"),
]


DATASET_LABELS = {
    "drive": "DRIVE",
    "chase_db1": "CHASE_DB1",
    "fives": "FIVES",
    "arcade_syntax": "ARCADE syntax",
    "arcade_stenosis": "ARCADE stenosis",
    "global_roads": "Global-Scale Roads",
    "gis_roads": "Roads_HF",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} +- {stdev(values):.4f}"


def _dataset_name(record: dict[str, Any], path: Path) -> str:
    raw = str(record.get("dataset") or "").lower()
    if raw:
        return raw
    text = str(path).lower()
    for key in DATASET_LABELS:
        if key in text:
            return key
    return "unknown"


def collect_records(results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in results_dir.rglob("metrics.json"):
        data = _read_json(metrics_path)
        if not data:
            continue
        if data.get("task") not in {None, "segmentation"}:
            continue
        if data.get("test_dice") is None:
            continue
        row = dict(data)
        row["_metrics_path"] = str(metrics_path)
        row["_dataset"] = _dataset_name(row, metrics_path)
        row["_variant"] = str(row.get("variant") or "unknown")
        records.append(row)
    return records


def build_summary(results_dir: Path, out_path: Path, include_gis: bool) -> None:
    records = collect_records(results_dir)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["_dataset"], row["_variant"])].append(row)

    lines: list[str] = []
    lines.append("# Article Results Summary")
    lines.append("")
    lines.append(f"Source directory: `{results_dir}`")
    lines.append("")
    lines.append("## Segmentation Metrics")
    lines.append("")
    header = ["Dataset", "Variant", "Runs", "Seeds", "Best run"] + [label for _, label in METRICS] + [
        "Threshold",
        "Params",
        "GMACs",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for dataset, variant in sorted(grouped):
        if not include_gis and dataset in {"global_roads", "gis_roads"}:
            continue
        rows = grouped[(dataset, variant)]
        best = max(rows, key=lambda item: float(item.get("test_dice") or -1.0))
        seeds = sorted({str(row.get("seed")) for row in rows if row.get("seed") is not None})
        metric_cells = []
        for key, _label in METRICS:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            metric_cells.append(_fmt(values))
        thresholds = [float(row["selected_threshold"]) for row in rows if row.get("selected_threshold") is not None]
        params = best.get("num_parameters")
        gmacs = best.get("approx_gmacs_per_forward")
        cells = [
            DATASET_LABELS.get(dataset, dataset),
            variant,
            str(len(rows)),
            ", ".join(seeds) if seeds else "-",
            str(best.get("run_name") or Path(str(best["_metrics_path"])).parent.name),
            *metric_cells,
            _fmt(thresholds),
            str(params) if params is not None else "-",
            f"{float(gmacs):.3f}" if gmacs is not None else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    if include_gis:
        lines.append("")
        lines.append("## GIS Note")
        lines.append("")
        lines.append(
            "GIS runs are included only as auxiliary evidence. The current article package treats medical "
            "segmentation as the main experimental block."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/article_final_latest")
    parser.add_argument("--out", default=None)
    parser.add_argument("--include-gis", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out) if args.out else results_dir / "article_results_summary_ru.md"
    build_summary(results_dir=results_dir, out_path=out_path, include_gis=bool(args.include_gis))
    print(out_path)


if __name__ == "__main__":
    main()
