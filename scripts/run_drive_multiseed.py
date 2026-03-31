#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
import utils
from utils import (
    ARTICLE_DRIVE_VARIANTS,
    CANONICAL_DRIVE_VARIANTS,
    compare_drive_metrics_to_baseline,
    ensure_dir,
    format_drive_superiority_report,
    load_config,
    save_json,
    update_drive_comparison_summary,
    update_drive_multiseed_summary,
    update_segmentation_multiseed_summary,
)


def _parse_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_seeds(text: str) -> List[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def _ordered_variants(variants: Sequence[str]) -> List[str]:
    items = list(variants)
    if "baseline" in items:
        return ["baseline"] + [variant for variant in items if variant != "baseline"]
    return items


def _load_variant_overrides(path: str | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    raw_path = Path(path)
    with open(raw_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Variant overrides file must contain a mapping, got {type(data).__name__}.")
    if "variant_overrides" in data:
        data = data["variant_overrides"]
    if not isinstance(data, dict):
        raise ValueError("variant_overrides must be a mapping of variant -> override dict.")
    overrides: Dict[str, Dict[str, Any]] = {}
    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValueError(f"Overrides for variant '{key}' must be a mapping.")
        overrides[str(key)] = dict(value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical DRIVE segmentation experiments across multiple seeds.")
    parser.add_argument("--config", type=str, default="configs/drive_benchmark.yaml")
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Comma-separated variants. Default: canonical DRIVE set.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="41,42,43",
        help="Comma-separated random seeds.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Session directory name under results_dir.")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results directory from config.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for all runs.")
    parser.add_argument(
        "--variant-overrides",
        type=str,
        default=None,
        help="Optional YAML file mapping variants to config overrides for the final benchmark.",
    )
    parser.add_argument(
        "--article-drive",
        action="store_true",
        help="Run the extended article comparison set: baseline, attention_unet, az_no_fuzzy, az_no_aniso, az_cat, az_thesis.",
    )
    parser.add_argument(
        "--require-beats-baseline",
        action="store_true",
        help="Fail if any non-baseline run does not beat the best stored baseline in the session directory.",
    )
    parser.add_argument(
        "--comparison-metrics",
        type=str,
        default=None,
        help="Comma-separated gated metrics. Default: DRIVE core metrics.",
    )
    parser.add_argument(
        "--comparison-min-delta",
        type=float,
        default=0.0,
        help="Required positive delta over baseline for each gated metric.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.results_dir is not None:
        cfg["results_dir"] = args.results_dir
    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)

    if args.variants:
        variants = _ordered_variants(_parse_csv(args.variants))
    elif args.article_drive:
        variants = _ordered_variants(list(ARTICLE_DRIVE_VARIANTS))
    else:
        variants = _ordered_variants(list(CANONICAL_DRIVE_VARIANTS))
    invalid = [variant for variant in variants if variant not in utils.VARIANTS]
    if invalid:
        raise SystemExit(f"Unknown variant(s) {invalid}. Choose from {utils.VARIANTS}")

    seeds = _parse_seeds(args.seeds)
    metric_names = (
        _parse_csv(args.comparison_metrics)
        if args.comparison_metrics
        else list(utils.DRIVE_SUPERIORITY_METRICS)
    )
    variant_overrides = _load_variant_overrides(args.variant_overrides)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = args.run_name or f"drive_benchmark_{timestamp}"
    session_dir = ensure_dir(Path(cfg.get("results_dir", "./results")) / session_name)
    dataset_name = str(cfg.get("dataset", "drive"))
    is_drive_dataset = utils._normalize_dataset_name(dataset_name) == "drive"

    all_metrics: List[Dict[str, Any]] = []
    failed_variants: List[str] = []

    print(f"Session directory: {session_dir}")
    print(f"Variants: {', '.join(variants)}")
    print(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    if variant_overrides:
        print(f"Variant overrides: {', '.join(sorted(variant_overrides))}")

    for seed in seeds:
        for variant in variants:
            run_name = f"{variant}_seed{seed}"
            run_dir = ensure_dir(session_dir / run_name)
            run_cfg = dict(cfg)
            run_cfg["seed"] = int(seed)
            run_cfg["variant"] = variant
            run_cfg["run_name"] = run_name
            if variant in variant_overrides:
                run_cfg.update(variant_overrides[variant])
                run_cfg["variant"] = variant
                run_cfg["run_name"] = run_name
                run_cfg["seed"] = int(seed)
            if args.device is not None:
                run_cfg["device"] = args.device
            if args.epochs is not None:
                run_cfg["epochs"] = int(args.epochs)

            print(f"[seed {seed}] variant={variant} -> {run_dir}")
            metrics = train.run_training(run_cfg, variant, run_dir)
            all_metrics.append(metrics)

            if args.require_beats_baseline and variant != "baseline":
                try:
                    report = compare_drive_metrics_to_baseline(
                        results_dir=session_dir,
                        candidate_metrics=metrics,
                        metric_names=metric_names,
                        min_delta=float(args.comparison_min_delta),
                    )
                except ValueError as exc:
                    raise SystemExit(f"Cannot evaluate baseline gate for {variant} seed={seed}: {exc}") from exc
                print(format_drive_superiority_report(report))
                if not report["all_passed"]:
                    failed_variants.append(f"{variant}@seed{seed}")

    save_json(session_dir / "all_metrics.json", all_metrics)
    comparison_path = None
    if is_drive_dataset:
        comparison_path = update_drive_comparison_summary(session_dir)
        multiseed_path = update_drive_multiseed_summary(session_dir, variants=variants)
    else:
        multiseed_path = update_segmentation_multiseed_summary(
            session_dir,
            dataset=dataset_name,
            variants=variants,
        )

    print(f"Wrote per-run metrics to {session_dir / 'all_metrics.json'}")
    if comparison_path is not None:
        print(f"Wrote best-run comparison summary to {comparison_path}")
    print(f"Wrote multi-seed summary to {multiseed_path}")

    if failed_variants:
        raise SystemExit("Baseline superiority gate failed for: " + ", ".join(failed_variants))


if __name__ == "__main__":
    main()
