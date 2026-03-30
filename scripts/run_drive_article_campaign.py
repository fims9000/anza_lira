#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(command: Sequence[str], workdir: Path) -> None:
    printable = " ".join(command)
    print(f"\n$ {printable}")
    subprocess.run(list(command), check=True, cwd=workdir)


def _drive_root_from_config(config_path: Path) -> Path:
    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    data_root = Path(cfg.get("data_root", "./data"))
    if data_root.is_absolute():
        return data_root / "DRIVE"
    return (PROJECT_ROOT / data_root / "DRIVE").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full DRIVE article campaign: az_thesis sweep -> best-trial promotion -> final multi-seed benchmark."
    )
    parser.add_argument("--sweep-config", type=str, default="configs/drive_thesis_tuning.yaml")
    parser.add_argument("--benchmark-config", type=str, default="configs/drive_benchmark.yaml")
    parser.add_argument("--seeds", type=str, default="41,42,43", help="Comma-separated random seeds for both sweep and benchmark.")
    parser.add_argument("--device", type=str, default="cuda", help="Device override passed to the child scripts.")
    parser.add_argument("--results-dir", type=str, default="results", help="Results root under the project.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional campaign directory name under results.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional hard limit for the thesis sweep.")
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Reuse an existing sweep directory instead of launching a new sweep run.",
    )
    parser.add_argument(
        "--sweep-session",
        type=str,
        default=None,
        help="Existing sweep session directory. Required when --skip-sweep is used.",
    )
    args = parser.parse_args()

    sweep_config = (PROJECT_ROOT / args.sweep_config).resolve()
    benchmark_config = (PROJECT_ROOT / args.benchmark_config).resolve()
    drive_root = _drive_root_from_config(sweep_config)
    if not drive_root.exists():
        raise SystemExit(f"DRIVE dataset is missing: expected {drive_root}")

    results_root = (PROJECT_ROOT / args.results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    campaign_name = args.run_name or f"drive_article_campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    campaign_dir = results_root / campaign_name
    campaign_dir.mkdir(parents=True, exist_ok=True)

    sweep_dir: Path
    if args.skip_sweep:
        if args.sweep_session is None:
            raise SystemExit("--sweep-session must be provided when --skip-sweep is used.")
        sweep_dir = Path(args.sweep_session)
        if not sweep_dir.is_absolute():
            sweep_dir = (PROJECT_ROOT / sweep_dir).resolve()
        if not sweep_dir.exists():
            raise SystemExit(f"Sweep session does not exist: {sweep_dir}")
    else:
        sweep_dir = campaign_dir / "sweep"
        command = [
            sys.executable,
            str((PROJECT_ROOT / "scripts" / "run_az_thesis_sweep.py").resolve()),
            "--config",
            str(sweep_config),
            "--seeds",
            args.seeds,
            "--device",
            args.device,
            "--results-dir",
            str(campaign_dir),
            "--run-name",
            "sweep",
        ]
        if args.max_trials is not None:
            command.extend(["--max-trials", str(int(args.max_trials))])
        _run(command, PROJECT_ROOT)

    overrides_path = sweep_dir / "best_trial_overrides.yaml"
    if not overrides_path.exists():
        raise SystemExit(f"Best-trial override file was not found: {overrides_path}")

    benchmark_dir = campaign_dir / "benchmark"
    benchmark_command = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "run_drive_multiseed.py").resolve()),
        "--config",
        str(benchmark_config),
        "--device",
        args.device,
        "--seeds",
        args.seeds,
        "--results-dir",
        str(campaign_dir),
        "--run-name",
        "benchmark",
        "--article-drive",
        "--variant-overrides",
        str(overrides_path),
    ]
    _run(benchmark_command, PROJECT_ROOT)

    manifest: dict[str, Any] = {
        "campaign_dir": str(campaign_dir),
        "sweep_dir": str(sweep_dir),
        "benchmark_dir": str(benchmark_dir),
        "sweep_config": str(sweep_config),
        "benchmark_config": str(benchmark_config),
        "best_trial_overrides": str(overrides_path),
        "seeds": [int(item.strip()) for item in args.seeds.split(",") if item.strip()],
        "device": args.device,
    }
    with open(campaign_dir / "campaign_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nCampaign ready: {campaign_dir}")
    print(f"Sweep results: {sweep_dir}")
    print(f"Benchmark results: {benchmark_dir}")
    print(f"Manifest: {campaign_dir / 'campaign_manifest.json'}")


if __name__ == "__main__":
    main()
