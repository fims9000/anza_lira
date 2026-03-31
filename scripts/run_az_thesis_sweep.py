#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train
import utils
from utils import ensure_dir, load_config, save_json, update_drive_comparison_summary, update_drive_multiseed_summary


def _parse_ints(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_floats(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _parse_strings(text: str) -> List[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one string value.")
    return values


def _parse_mixed_auto_floats(text: str) -> List[str | float]:
    values: List[str | float] = []
    for item in text.split(","):
        token = item.strip()
        if not token:
            continue
        if token.lower() == "auto":
            values.append("auto")
        else:
            values.append(float(token))
    if not values:
        raise ValueError("Expected at least one value, e.g. 'auto,8,10'.")
    return values


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    tensor = np.asarray(list(values), dtype=np.float64)
    return float(tensor.mean()), float(tensor.std(ddof=0))


def _slug_float(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _display_scalar(value: str | float | int) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):g}"


def _parse_width_sets(text: str) -> List[tuple[int, int, int, int]]:
    width_sets: List[tuple[int, int, int, int]] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        width_sets.append(utils.parse_model_widths(chunk))
    if not width_sets:
        raise ValueError("Expected at least one width set, e.g. '48,96,160,224;40,80,128,192'.")
    return width_sets


def _slug_widths(widths: Sequence[int]) -> str:
    return "x".join(str(int(width)) for width in widths)


def _trial_name(
    num_rules: int,
    widths: Sequence[int],
    lr: float,
    encoder_az_stages: int,
    hybrid_mix_init: float,
    boundary_weight: float,
    topology_weight: float,
    overlap_mode: str,
    tversky_alpha: float,
    tversky_beta: float,
    bce_pos_weight: str | float,
    threshold_metric: str,
    bottleneck_mode: str,
    decoder_mode: str,
    boundary_mode: str,
) -> str:
    pos_weight_slug = str(bce_pos_weight) if isinstance(bce_pos_weight, str) else _slug_float(float(bce_pos_weight))
    return (
        f"r{num_rules}_lr{_slug_float(lr)}"
        f"_w{_slug_widths(widths)}"
        f"_enc{int(encoder_az_stages)}"
        f"_hm{_slug_float(hybrid_mix_init)}"
        f"_bw{_slug_float(boundary_weight)}"
        f"_tw{_slug_float(topology_weight)}"
        f"_ov{overlap_mode}"
        f"_ta{_slug_float(tversky_alpha)}"
        f"_tb{_slug_float(tversky_beta)}"
        f"_pw{pos_weight_slug}"
        f"_tm{threshold_metric}"
        f"_bn{bottleneck_mode}"
        f"_dec{decoder_mode}"
        f"_head{boundary_mode}"
    )


def _ranking_row(trial_name: str, trial_cfg: Dict[str, Any], metrics_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trial_name": trial_name,
        "num_runs": len(metrics_rows),
        "seeds": [int(item["seed"]) for item in metrics_rows if item.get("seed") is not None],
        "num_rules": int(trial_cfg["num_rules"]),
        "model_widths": list(utils.parse_model_widths(trial_cfg.get("model_widths")) or []),
        "lr": float(trial_cfg["lr"]),
        "encoder_az_stages": int(trial_cfg["encoder_az_stages"]),
        "encoder_block_mode": str(trial_cfg.get("encoder_block_mode", "az")),
        "hybrid_mix_init": float(trial_cfg.get("hybrid_mix_init", 0.5)),
        "boundary_loss_weight": float(trial_cfg["boundary_loss_weight"]),
        "topology_loss_weight": float(trial_cfg["topology_loss_weight"]),
        "overlap_mode": str(trial_cfg["overlap_mode"]),
        "tversky_alpha": float(trial_cfg["tversky_alpha"]),
        "tversky_beta": float(trial_cfg["tversky_beta"]),
        "bce_pos_weight": trial_cfg["bce_pos_weight"],
        "eval_threshold_metric": str(trial_cfg["eval_threshold_metric"]),
        "bottleneck_mode": str(trial_cfg["bottleneck_mode"]),
        "decoder_mode": str(trial_cfg["decoder_mode"]),
        "boundary_mode": str(trial_cfg["boundary_mode"]),
        "epochs": int(trial_cfg["epochs"]),
    }
    for metric in utils.DRIVE_MULTI_SEED_METRICS:
        values = [float(item[metric]) for item in metrics_rows if item.get(metric) is not None]
        mean_value, std_value = _mean_std(values)
        row[f"{metric}_mean"] = mean_value
        row[f"{metric}_std"] = std_value
    return row


def _write_markdown_ranking(out_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "| Trial | Runs | Rules | Widths | Enc AZ | Hybrid mix | LR | Boundary | Topology | Overlap | Tversky a | Tversky b | BCE pos | Threshold metric | Bottleneck | Decoder | Boundary head | Dice mean+-std | IoU mean+-std | Precision mean+-std | Recall mean+-std | Specificity mean+-std | Balanced Acc mean+-std | Threshold mean+-std | Fwd mean (s) |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["trial_name"],
                    str(int(row["num_runs"])),
                    str(int(row["num_rules"])),
                    ",".join(str(int(width)) for width in row["model_widths"]) or "default",
                    str(int(row["encoder_az_stages"])),
                    f"{row['hybrid_mix_init']:.3f}",
                    f"{row['lr']:.6f}",
                    f"{row['boundary_loss_weight']:.4f}",
                    f"{row['topology_loss_weight']:.4f}",
                    row["overlap_mode"],
                    f"{row['tversky_alpha']:.3f}",
                    f"{row['tversky_beta']:.3f}",
                    _display_scalar(row["bce_pos_weight"]),
                    row["eval_threshold_metric"],
                    row["bottleneck_mode"],
                    row["decoder_mode"],
                    row["boundary_mode"],
                    f"{row['test_dice_mean']:.4f} +- {row['test_dice_std']:.4f}",
                    f"{row['test_iou_mean']:.4f} +- {row['test_iou_std']:.4f}",
                    f"{row['test_precision_mean']:.4f} +- {row['test_precision_std']:.4f}",
                    f"{row['test_recall_mean']:.4f} +- {row['test_recall_std']:.4f}",
                    f"{row['test_specificity_mean']:.4f} +- {row['test_specificity_std']:.4f}",
                    f"{row['test_balanced_accuracy_mean']:.4f} +- {row['test_balanced_accuracy_std']:.4f}",
                    f"{row['selected_threshold_mean']:.4f} +- {row['selected_threshold_std']:.4f}",
                    f"{row['seconds_per_forward_batch_mean']:.5f}",
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best_trial_artifacts(
    base_cfg: Dict[str, Any],
    best_row: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    overrides = {
        "num_rules": int(best_row["num_rules"]),
        "model_widths": list(best_row["model_widths"]),
        "lr": float(best_row["lr"]),
        "encoder_az_stages": int(best_row["encoder_az_stages"]),
        "encoder_block_mode": str(best_row.get("encoder_block_mode", "az")),
        "hybrid_mix_init": float(best_row.get("hybrid_mix_init", 0.5)),
        "boundary_loss_weight": float(best_row["boundary_loss_weight"]),
        "topology_loss_weight": float(best_row["topology_loss_weight"]),
        "overlap_mode": str(best_row["overlap_mode"]),
        "tversky_alpha": float(best_row["tversky_alpha"]),
        "tversky_beta": float(best_row["tversky_beta"]),
        "bce_pos_weight": best_row["bce_pos_weight"],
        "eval_threshold_metric": str(best_row["eval_threshold_metric"]),
        "bottleneck_mode": str(best_row["bottleneck_mode"]),
        "decoder_mode": str(best_row["decoder_mode"]),
        "boundary_mode": str(best_row["boundary_mode"]),
        "epochs": int(best_row["epochs"]),
    }
    full_cfg = dict(base_cfg)
    full_cfg.update(overrides)
    full_cfg["variant"] = "az_thesis"
    metadata = {
        "trial_name": best_row["trial_name"],
        "selection_metrics": {
            "test_dice_mean": float(best_row["test_dice_mean"]),
            "test_iou_mean": float(best_row["test_iou_mean"]),
            "test_balanced_accuracy_mean": float(best_row["test_balanced_accuracy_mean"]),
            "seconds_per_forward_batch_mean": float(best_row["seconds_per_forward_batch_mean"]),
        },
        "seeds": list(best_row["seeds"]),
    }
    return overrides, full_cfg, metadata


def _iter_trials(
    num_rules_values: Sequence[int],
    width_sets: Sequence[Sequence[int]],
    learning_rates: Sequence[float],
    encoder_az_stages_values: Sequence[int],
    hybrid_mix_values: Sequence[float],
    boundary_weights: Sequence[float],
    topology_weights: Sequence[float],
    overlap_modes: Sequence[str],
    tversky_alphas: Sequence[float],
    tversky_betas: Sequence[float],
    bce_pos_weights: Sequence[str | float],
    threshold_metrics: Sequence[str],
    bottleneck_modes: Sequence[str],
    decoder_modes: Sequence[str],
    boundary_modes: Sequence[str],
) -> Iterable[tuple[Any, ...]]:
    return itertools.product(
        num_rules_values,
        width_sets,
        learning_rates,
        encoder_az_stages_values,
        hybrid_mix_values,
        boundary_weights,
        topology_weights,
        overlap_modes,
        tversky_alphas,
        tversky_betas,
        bce_pos_weights,
        threshold_metrics,
        bottleneck_modes,
        decoder_modes,
        boundary_modes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed hyperparameter sweep for the az_thesis DRIVE model.")
    parser.add_argument("--config", type=str, default="configs/drive_thesis_tuning.yaml")
    parser.add_argument("--seeds", type=str, default="41,42,43", help="Comma-separated random seeds.")
    parser.add_argument("--num-rules", type=str, default="4,6,8", help="Comma-separated ANZA rule counts.")
    parser.add_argument(
        "--widths",
        type=str,
        default="32,64,128,192;40,80,144,208;48,96,160,224",
        help="Semicolon-separated width sets, each written as four comma-separated integers.",
    )
    parser.add_argument("--learning-rates", type=str, default="0.0003,0.0002,0.0001", help="Comma-separated learning rates.")
    parser.add_argument("--encoder-az-stages", type=str, default="1,2,3", help="Comma-separated counts of encoder stages that use AZ blocks.")
    parser.add_argument("--hybrid-mix-inits", type=str, default="0.5", help="Comma-separated initial AZ-branch mixture values for hybrid encoder blocks.")
    parser.add_argument("--boundary-weights", type=str, default="0.05,0.1,0.15", help="Comma-separated boundary loss weights.")
    parser.add_argument("--topology-weights", type=str, default="0.0,0.01,0.03", help="Comma-separated topology loss weights.")
    parser.add_argument("--overlap-modes", type=str, default="dice,tversky", help="Comma-separated overlap modes.")
    parser.add_argument("--tversky-alphas", type=str, default="0.5", help="Comma-separated Tversky alpha values.")
    parser.add_argument("--tversky-betas", type=str, default="0.5", help="Comma-separated Tversky beta values.")
    parser.add_argument("--bce-pos-weights", type=str, default="auto", help="Comma-separated BCE positive weights or 'auto'.")
    parser.add_argument("--threshold-metrics", type=str, default="dice,core_mean", help="Comma-separated validation threshold metrics.")
    parser.add_argument("--bottleneck-modes", type=str, default="az_single", help="Comma-separated thesis bottleneck modes.")
    parser.add_argument("--decoder-modes", type=str, default="residual", help="Comma-separated thesis decoder modes.")
    parser.add_argument("--boundary-modes", type=str, default="conv", help="Comma-separated thesis boundary head modes.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for every trial.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda or cpu.")
    parser.add_argument("--results-dir", type=str, default=None, help="Optional results root override.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional session directory name.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional hard limit on the number of grid trials.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    if args.device is not None:
        cfg["device"] = str(args.device)
    if args.results_dir is not None:
        cfg["results_dir"] = str(args.results_dir)
    cfg["variant"] = "az_thesis"

    seeds = _parse_ints(args.seeds)
    num_rules_values = _parse_ints(args.num_rules)
    width_sets = _parse_width_sets(args.widths)
    learning_rates = _parse_floats(args.learning_rates)
    encoder_az_stages_values = _parse_ints(args.encoder_az_stages)
    hybrid_mix_values = _parse_floats(args.hybrid_mix_inits)
    boundary_weights = _parse_floats(args.boundary_weights)
    topology_weights = _parse_floats(args.topology_weights)
    overlap_modes = _parse_strings(args.overlap_modes)
    tversky_alphas = _parse_floats(args.tversky_alphas)
    tversky_betas = _parse_floats(args.tversky_betas)
    bce_pos_weights = _parse_mixed_auto_floats(args.bce_pos_weights)
    threshold_metrics = _parse_strings(args.threshold_metrics)
    bottleneck_modes = _parse_strings(args.bottleneck_modes)
    decoder_modes = _parse_strings(args.decoder_modes)
    boundary_modes = _parse_strings(args.boundary_modes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = args.run_name or f"az_thesis_sweep_{timestamp}"
    session_dir = ensure_dir(Path(cfg.get("results_dir", "./results")) / session_name)

    all_metrics: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []

    all_trials = list(
        _iter_trials(
            num_rules_values=num_rules_values,
            width_sets=width_sets,
            learning_rates=learning_rates,
            encoder_az_stages_values=encoder_az_stages_values,
            hybrid_mix_values=hybrid_mix_values,
            boundary_weights=boundary_weights,
            topology_weights=topology_weights,
            overlap_modes=overlap_modes,
            tversky_alphas=tversky_alphas,
            tversky_betas=tversky_betas,
            bce_pos_weights=bce_pos_weights,
            threshold_metrics=threshold_metrics,
            bottleneck_modes=bottleneck_modes,
            decoder_modes=decoder_modes,
            boundary_modes=boundary_modes,
        )
    )
    if args.max_trials is not None:
        all_trials = all_trials[: max(int(args.max_trials), 0)]

    print(f"Session directory: {session_dir}")
    print(f"Seeds: {', '.join(str(seed) for seed in seeds)}")
    print(f"Trials: {len(all_trials)}")

    for trial_index, trial in enumerate(all_trials, start=1):
        (
            num_rules,
            widths,
            lr,
            encoder_az_stages,
            hybrid_mix_init,
            boundary_weight,
            topology_weight,
            overlap_mode,
            tversky_alpha,
            tversky_beta,
            bce_pos_weight,
            threshold_metric,
            bottleneck_mode,
            decoder_mode,
            boundary_mode,
        ) = trial
        trial_name = _trial_name(
            num_rules=num_rules,
            widths=widths,
            lr=lr,
            encoder_az_stages=encoder_az_stages,
            hybrid_mix_init=hybrid_mix_init,
            boundary_weight=boundary_weight,
            topology_weight=topology_weight,
            overlap_mode=overlap_mode,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            bce_pos_weight=bce_pos_weight,
            threshold_metric=threshold_metric,
            bottleneck_mode=bottleneck_mode,
            decoder_mode=decoder_mode,
            boundary_mode=boundary_mode,
        )
        trial_dir = ensure_dir(session_dir / trial_name)
        trial_cfg = dict(cfg)
        trial_cfg.update(
            {
                "variant": "az_thesis",
                "num_rules": int(num_rules),
                "model_widths": list(widths),
                "lr": float(lr),
                "encoder_az_stages": int(encoder_az_stages),
                "hybrid_mix_init": float(hybrid_mix_init),
                "boundary_loss_weight": float(boundary_weight),
                "topology_loss_weight": float(topology_weight),
                "overlap_mode": str(overlap_mode),
                "tversky_alpha": float(tversky_alpha),
                "tversky_beta": float(tversky_beta),
                "bce_pos_weight": bce_pos_weight,
                "eval_threshold_metric": str(threshold_metric),
                "bottleneck_mode": str(bottleneck_mode),
                "decoder_mode": str(decoder_mode),
                "boundary_mode": str(boundary_mode),
            }
        )

        print(
            f"[trial {trial_index}/{len(all_trials)}] {trial_name} "
            f"(rules={num_rules}, widths={list(widths)}, enc_az={encoder_az_stages}, hm={hybrid_mix_init}, lr={lr}, boundary={boundary_weight}, topology={topology_weight}, "
            f"overlap={overlap_mode}, tversky=({tversky_alpha},{tversky_beta}), pos_weight={_display_scalar(bce_pos_weight)}, "
            f"thr_metric={threshold_metric}, bottleneck={bottleneck_mode}, "
            f"decoder={decoder_mode}, head={boundary_mode})"
        )

        trial_metrics: List[Dict[str, Any]] = []
        for seed in seeds:
            run_name = f"az_thesis_seed{seed}"
            run_dir = ensure_dir(trial_dir / run_name)
            run_cfg = dict(trial_cfg)
            run_cfg["seed"] = int(seed)
            run_cfg["run_name"] = run_name
            metrics = train.run_training(run_cfg, "az_thesis", run_dir)
            trial_metrics.append(metrics)
            all_metrics.append(metrics)

        update_drive_comparison_summary(trial_dir)
        update_drive_multiseed_summary(trial_dir, variants=["az_thesis"])
        ranking_rows.append(_ranking_row(trial_name, trial_cfg, trial_metrics))

    ranking_rows = sorted(
        ranking_rows,
        key=lambda row: (
            -float(row["test_dice_mean"]),
            -float(row["test_balanced_accuracy_mean"]),
            -float(row["test_iou_mean"]),
            float(row["seconds_per_forward_batch_mean"]),
        ),
    )

    save_json(session_dir / "all_metrics.json", all_metrics)
    save_json(session_dir / "trial_ranking.json", ranking_rows)
    _write_markdown_ranking(session_dir / "trial_ranking.md", ranking_rows)

    if ranking_rows:
        best_row = ranking_rows[0]
        best_overrides, best_cfg, best_meta = _best_trial_artifacts(cfg, best_row)
        save_json(session_dir / "best_trial.json", best_row)
        save_json(session_dir / "best_trial_overrides.json", {"variant_overrides": {"az_thesis": best_overrides}, "metadata": best_meta})
        with open(session_dir / "best_trial_overrides.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump({"variant_overrides": {"az_thesis": best_overrides}, "metadata": best_meta}, handle, sort_keys=False)
        with open(session_dir / "best_trial_config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(best_cfg, handle, sort_keys=False)

    print(f"Wrote run-level metrics to {session_dir / 'all_metrics.json'}")
    print(f"Wrote trial ranking JSON to {session_dir / 'trial_ranking.json'}")
    print(f"Wrote trial ranking markdown to {session_dir / 'trial_ranking.md'}")
    if ranking_rows:
        print(f"Wrote best-trial summary to {session_dir / 'best_trial.json'}")
        print(f"Wrote best-trial overrides to {session_dir / 'best_trial_overrides.yaml'}")
        print(f"Wrote full best-trial config to {session_dir / 'best_trial_config.yaml'}")


if __name__ == "__main__":
    main()
