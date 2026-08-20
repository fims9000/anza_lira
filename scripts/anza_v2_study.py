#!/usr/bin/env python3
"""Resumable ANZA-LIRA v2 study entry point."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _bootstrap_full_data_contract_before_imports() -> None:
    """Let ``full`` start from verified ZIPs before artifact-bound imports."""
    if len(sys.argv) < 2 or sys.argv[1] != "full":
        return
    prerequisites = (
        (
            PROJECT_ROOT / "results" / "cracks_study" / "archive_inventory.json",
            [sys.executable, "scripts/audit_cracks_archives.py"],
        ),
        (
            PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json",
            [sys.executable, "scripts/prepare_cracks_protocol.py"],
        ),
        (
            PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "crowd_target" / "manifest.json",
            [sys.executable, "scripts/build_cracks_crowd_targets.py"],
        ),
        (
            PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "crowd_target" / "normalization.json",
            [sys.executable, "scripts/compute_cracks_normalization.py"],
        ),
    )
    for artifact, command in prerequisites:
        if artifact.exists():
            continue
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)


_bootstrap_full_data_contract_before_imports()

from synthetic.experiment_matrix import COMMON_PROTOCOL, development_matrix, protocol_hash
from synthetic.evaluation import evaluate_candidate, evaluate_frozen_test, evaluation_protocol_hash
from synthetic.diagnostics import diagnose_candidate
from synthetic.quality_gate import freeze_validation_candidate
from synthetic.training import run_candidate_development, run_candidate_smoke
from cracks_experiment.matrix import setting_a_matrix, setting_a_protocol_hash
from cracks_experiment.training import run_setting_a_training
from cracks_experiment.validation import freeze_setting_a_thresholds, run_setting_a_validation
from cracks_experiment.evaluation import finalize_setting_a_expert_evaluation, run_setting_a_expert_evaluation
from cracks_experiment.finetuning import FOLDS, run_setting_b_fold, setting_b_sources
from cracks_experiment.robustness import run_setting_c_fold, setting_c_models
from cracks_experiment.human import run_disagreement_analysis, run_human_baseline
from cracks_experiment.statistics import build_statistics
from cracks_experiment.efficiency import run_efficiency_audit
from cracks_experiment.evidence import build_thesis_evidence
from cracks_experiment.figures import generate_figures
from cracks_experiment.traces import export_setting_a_traces


def _dry_run() -> int:
    print("ANZA-LIRA V2 SYNTHETIC DEVELOPMENT MATRIX")
    for spec in development_matrix():
        print(
            f"{spec.candidate_id:<2}  {spec.model:<18} seed={spec.seed} "
            f"objectives={'+'.join(spec.objectives)} run={spec.run_hash}"
        )
    print(f"PROTOCOL HASH: {protocol_hash()}")
    print("SYNTHETIC TEST: FROZEN_UNOPENED")
    return 0


def _write_protocol() -> int:
    output = PROJECT_ROOT / "results" / "anza_v2_study" / "synthetic" / "development_matrix.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PREPARED",
        "scientific_result": False,
        "protocol_hash": protocol_hash(),
        "common_protocol": COMMON_PROTOCOL,
        "runs": [
            {
                "candidate_id": spec.candidate_id,
                "model": spec.model,
                "objectives": list(spec.objectives),
                "comparison_family": spec.comparison_family,
                "seed": spec.seed,
                "kappa_theta": spec.kappa_theta,
                "kappa_direction": spec.kappa_direction,
                "run_hash": spec.run_hash,
            }
            for spec in development_matrix()
        ],
        "dsc_unet": "NOT_INCLUDED_DEPENDENCY_SCOPE",
        "deformable_backend": "native_torch_grid_sample; torchvision import unavailable",
        "test_stream": "FROZEN_UNOPENED",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"WROTE: {output}")
    print(f"PROTOCOL HASH: {payload['protocol_hash']}")
    return 0


def _smoke() -> int:
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "synthetic" / "smoke"
    for spec in development_matrix():
        result = run_candidate_smoke(spec, output_root, epochs=1, image_size=16)
        print(f"{spec.candidate_id} {spec.model}: {result['status']} ({result['action']})")
    print("SYNTHETIC TRAIN/CHECKPOINT SMOKE: PASS")
    print("SCIENTIFIC RESULT: FALSE")
    print("SYNTHETIC TEST: FROZEN_UNOPENED")
    return 0


def _develop() -> int:
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "synthetic" / "development"
    for spec in development_matrix():
        result = run_candidate_development(spec, output_root)
        print(f"{spec.candidate_id} {spec.model}: {result['status']} ({result['action']})")
    print("SYNTHETIC DEVELOPMENT TRAINING: COMPLETE")
    print("STRUCTURAL EVALUATION: WAITING")
    print("SYNTHETIC TEST: FROZEN_UNOPENED")
    return 0


def _evaluate() -> int:
    development_root = PROJECT_ROOT / "results" / "anza_v2_study" / "synthetic" / "development"
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "synthetic" / "validation"
    for spec in development_matrix():
        result = evaluate_candidate(spec, development_root, output_root)
        metrics = result["metrics"]
        print(
            f"model={spec.candidate_id} visible_dice={metrics['visible_dice']:.4f} "
            f"pairing={metrics['branch_pairing_accuracy']:.4f} "
            f"false_merge={metrics['false_merge_rate']:.4f} "
            f"gap_recovery={metrics['gap_recovery_rate']:.4f} "
            f"false_bridge={metrics['false_bridge_rate']:.4f} status=COMPLETE"
        )
    print(f"EVALUATION PROTOCOL HASH: {evaluation_protocol_hash()}")
    print("SYNTHETIC VALIDATION: COMPLETE")
    print("SYNTHETIC TEST: FROZEN_UNOPENED")
    return 0


def _diagnose() -> int:
    development_root = PROJECT_ROOT / "results" / "anza_v2_study" / "synthetic" / "development"
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "synthetic" / "diagnostics"
    for spec in development_matrix():
        if not spec.model.startswith("anza_v2"):
            continue
        result = diagnose_candidate(spec, development_root, output_root)
        regions = result["regions"]
        print(
            f"model={spec.candidate_id} route_top1={result['route_top1_accuracy']:.4f} "
            f"route_confidence={result['route_max_probability']:.4f} "
            f"neff_junction={regions['junction']['effective_modes']} "
            f"neff_straight={regions['straight']['effective_modes']} "
            f"completion_background={result['completion_background_probability']}"
        )
    print("SYNTHETIC FAILURE LOCALIZATION: COMPLETE")
    print("SYNTHETIC TEST: FROZEN_UNOPENED")
    return 0


def _freeze() -> int:
    study_root = PROJECT_ROOT / "results" / "anza_v2_study"
    result = freeze_validation_candidate(study_root)
    print(f"SYNTHETIC QUALITY GATE: {result['quality_gate']}")
    print(f"FROZEN CANDIDATE: {result['frozen_candidate_id']} ({result['frozen_model']})")
    print(f"FREEZE SHA256: {result['freeze_sha256']}")
    print("SYNTHETIC TEST: AUTHORIZED_ONCE_NOT_OPENED")
    return 0


def _test() -> int:
    study_root = PROJECT_ROOT / "results" / "anza_v2_study"
    result = evaluate_frozen_test(study_root)
    print(f"SYNTHETIC TEST: {result['status']} ({result['action']})")
    print(f"FROZEN CANDIDATE: {result['frozen_candidate_id']}")
    return 0


def _cracks_dry_run() -> int:
    print("CRACKS SETTING A CROWD-ONLY MATRIX")
    for spec in setting_a_matrix():
        print(
            f"{spec.run_id:<24} model={spec.model:<18} seed={spec.seed} "
            f"replay={str(spec.structural_replay).lower()} run={spec.run_hash}"
        )
    print(f"SETTING A PROTOCOL HASH: {setting_a_protocol_hash()}")
    print("EXPERT SCORES: LOCKED")
    return 0


def _cracks_smoke() -> int:
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a_smoke"
    selected = [
        spec
        for spec in setting_a_matrix()
        if spec.comparison_family == "main" and spec.seed == 42
    ]
    for spec in selected:
        result = run_setting_a_training(
            spec,
            output_root,
            epochs=1,
            max_train_sections=4,
        )
        print(f"{spec.run_id}: {result['status']} ({result['action']}) expert=LOCKED")
    print("CRACKS SETTING A SMOKE: PASS")
    print("EXPERT SCORES: LOCKED")
    return 0


def _cracks_train() -> int:
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
    for spec in setting_a_matrix():
        result = run_setting_a_training(spec, output_root)
        print(f"{spec.run_id}: {result['status']} ({result['action']}) expert=LOCKED")
    print("CRACKS SETTING A CROWD TRAINING: COMPLETE")
    print("FULL HELDOUT VALIDATION/THRESHOLD: WAITING")
    print("EXPERT SCORES: LOCKED")
    return 0


def _cracks_validate() -> int:
    training_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
    for spec in setting_a_matrix():
        result = run_setting_a_validation(spec, training_root)
        print(
            f"{spec.run_id}: {result['status']} ({result['action']}) "
            f"threshold={result['selected_threshold']:.2f} expert=LOCKED"
        )
    print("CRACKS SETTING A CROWD VALIDATION/THRESHOLD: COMPLETE")
    receipt = freeze_setting_a_thresholds(training_root)
    print(f"SETTING A THRESHOLDS FROZEN: {receipt['freeze_sha256']}")
    print("EXPERT SCORES: LOCKED")
    return 0


def _cracks_expert() -> int:
    training_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a_expert"
    for spec in setting_a_matrix():
        result = run_setting_a_expert_evaluation(spec, training_root, output_root)
        summary = result["summary"]
        print(
            f"{spec.run_id}: {result['status']} ({result['action']}) "
            f"dice={summary['dice']:.4f} cldice={summary['cldice']:.4f}"
        )
    receipt = finalize_setting_a_expert_evaluation(training_root, output_root)
    print(f"SETTING A EXPERT RECEIPT: {receipt['sha256']}")
    print("CRACKS SETTING A EXPERT EVALUATION: COMPLETE")
    return 0


def _cracks_setting_b() -> int:
    setting_a_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
    expert_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a_expert"
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_b"
    for spec in setting_b_sources():
        for fold in FOLDS["folds"]:
            result = run_setting_b_fold(spec, fold, setting_a_root, expert_root, output_root)
            print(
                f"model={spec.model} fold={fold['fold']} {result['status']} ({result['action']}) "
                f"dice={result['summary']['dice']:.4f}"
            )
    print("CRACKS SETTING B LIMITED-EXPERT CV: COMPLETE")
    return 0


def _cracks_setting_c() -> int:
    setting_a_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
    expert_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a_expert"
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_c"
    for spec in setting_c_models():
        for fold in FOLDS["folds"]:
            result = run_setting_c_fold(spec, fold, setting_a_root, expert_root, output_root)
            print(
                f"model={spec.model} fold={fold['fold']} {result['status']} ({result['action']}) "
                f"dice={result['summary']['dice']:.4f}"
            )
    print("CRACKS SETTING C IMAGE-DISJOINT ROBUSTNESS: COMPLETE")
    return 0


def _cracks_human() -> int:
    setting_a_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
    expert_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a_expert"
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "human_comparison"
    result = run_human_baseline(setting_a_root, expert_root, output_root)
    print(
        f"CRACKS HUMAN BASELINE: {result['status']} ({result['action']}) "
        f"rows={result['row_count']}"
    )
    return 0


def _cracks_disagreement() -> int:
    setting_a_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
    expert_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a_expert"
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "disagreement"
    result = run_disagreement_analysis(setting_a_root, expert_root, output_root)
    print(f"CRACKS HUMAN DISAGREEMENT: {result['status']} ({result['action']})")
    return 0


def _statistics() -> int:
    result = build_statistics(PROJECT_ROOT / "results" / "anza_v2_study")
    print(
        f"ANZA-LIRA V2 STATISTICS: {result['status']} "
        f"bootstrap_unit={result['bootstrap_unit']} resamples={result['bootstrap_resamples']}"
    )
    return 0


def _efficiency() -> int:
    setting_a_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "efficiency"
    result = run_efficiency_audit(setting_a_root, output_root)
    print(f"ANZA-LIRA V2 EFFICIENCY: {result['status']} ({result['action']})")
    return 0


def _evidence() -> int:
    result = build_thesis_evidence(PROJECT_ROOT / "results" / "anza_v2_study")
    print(f"ANZA-LIRA V2 THESIS EVIDENCE: {result['status']} report={result['report_consistency']}")
    return 0


def _figures() -> int:
    result = generate_figures(PROJECT_ROOT / "results" / "anza_v2_study")
    print(f"ANZA-LIRA V2 FIGURES: {result['status']} ({result['action']})")
    return 0


def _cracks_traces() -> int:
    setting_a_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a"
    expert_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "setting_a_expert"
    output_root = PROJECT_ROOT / "results" / "anza_v2_study" / "cracks" / "traces"
    result = export_setting_a_traces(setting_a_root, expert_root, output_root)
    print(f"CRACKS CANDIDATE TRACES: {result['status']} ({result['action']}) count={result['geojson_count']}")
    return 0


def _run_logged_phase(name: str, command: list[str], log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{name}.log"
    print(f"phase={name} status=STARTED log={log_path}", flush=True)
    with log_path.open("a") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            compact = line.strip()
            if compact.startswith("phase=") or any(
                token in compact for token in (" COMPLETE", " PASS", " FAIL", "STATUS:")
            ):
                print(compact, flush=True)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Phase {name} failed with exit {return_code}; see {log_path}")
    print(f"phase={name} status=COMPLETE", flush=True)


def _full() -> int:
    study_root = PROJECT_ROOT / "results" / "anza_v2_study"
    log_root = study_root / "logs"
    python = sys.executable
    required_data = (
        PROJECT_ROOT / "results" / "cracks_study" / "archive_inventory.json",
        study_root / "protocol.json",
        study_root / "cracks" / "crowd_target" / "manifest.json",
        study_root / "cracks" / "crowd_target" / "normalization.json",
    )
    data_commands = (
        ("archive_audit", [python, "scripts/audit_cracks_archives.py"]),
        ("cracks_protocol", [python, "scripts/prepare_cracks_protocol.py"]),
        ("crowd_targets", [python, "scripts/build_cracks_crowd_targets.py"]),
        ("normalization", [python, "scripts/compute_cracks_normalization.py"]),
    )
    for path, (name, command) in zip(required_data, data_commands):
        if not path.exists():
            _run_logged_phase(name, command, log_root)
        else:
            print(f"phase={name} status=SKIP artifact={path}")

    synthetic_receipt = study_root / "synthetic" / "test" / "test_open_receipt.json"
    if not synthetic_receipt.exists():
        for command in ("prepare", "smoke", "develop", "evaluate", "diagnose", "freeze", "test"):
            _run_logged_phase(f"synthetic_{command}", [python, __file__, command], log_root)
    else:
        print(f"phase=synthetic status=SKIP receipt={synthetic_receipt}")

    for command in (
        "cracks-train",
        "cracks-validate",
        "cracks-expert",
        "cracks-setting-b",
        "cracks-setting-c",
        "cracks-human",
        "cracks-disagreement",
        "cracks-traces",
        "efficiency",
        "statistics",
        "figures",
        "evidence",
    ):
        _run_logged_phase(command.replace("-", "_"), [python, __file__, command], log_root)

    pytest_log = log_root / "pytest.log"
    started = datetime.now(timezone.utc).isoformat()
    result = subprocess.run(
        [python, "-m", "pytest"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    pytest_log.write_text(result.stdout)
    receipt = {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "exit_code": result.returncode,
        "command": f"{python} -m pytest",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(pytest_log),
    }
    (study_root / "pytest_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if result.returncode:
        raise RuntimeError(f"Full pytest failed; see {pytest_log}")
    print("phase=pytest status=PASS", flush=True)
    _run_logged_phase("final_validator", [python, "scripts/validate_anza_v2_study.py"], log_root)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "dry-run", "prepare", "smoke", "develop", "evaluate", "diagnose", "freeze", "test",
            "cracks-dry-run", "cracks-smoke", "cracks-train",
            "cracks-validate",
            "cracks-expert",
            "cracks-setting-b",
            "cracks-setting-c",
            "cracks-human",
            "cracks-disagreement",
            "cracks-traces",
            "statistics",
            "efficiency",
            "evidence",
            "figures",
            "full",
        ),
    )
    args = parser.parse_args()
    if args.command == "dry-run":
        return _dry_run()
    if args.command == "prepare":
        return _write_protocol()
    if args.command == "smoke":
        return _smoke()
    if args.command == "develop":
        return _develop()
    if args.command == "evaluate":
        return _evaluate()
    if args.command == "diagnose":
        return _diagnose()
    if args.command == "freeze":
        return _freeze()
    if args.command == "test":
        return _test()
    if args.command == "cracks-dry-run":
        return _cracks_dry_run()
    if args.command == "cracks-smoke":
        return _cracks_smoke()
    if args.command == "cracks-validate":
        return _cracks_validate()
    if args.command == "cracks-expert":
        return _cracks_expert()
    if args.command == "cracks-setting-b":
        return _cracks_setting_b()
    if args.command == "cracks-setting-c":
        return _cracks_setting_c()
    if args.command == "cracks-human":
        return _cracks_human()
    if args.command == "cracks-disagreement":
        return _cracks_disagreement()
    if args.command == "cracks-traces":
        return _cracks_traces()
    if args.command == "statistics":
        return _statistics()
    if args.command == "efficiency":
        return _efficiency()
    if args.command == "evidence":
        return _evidence()
    if args.command == "figures":
        return _figures()
    if args.command == "full":
        return _full()
    return _cracks_train()


if __name__ == "__main__":
    raise SystemExit(main())
