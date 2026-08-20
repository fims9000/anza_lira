"""Immutable receipts for the completed K1.5 factor control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/anza_ks/k1_5"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_k1_5() -> dict[str, object]:
    source_paths = sorted((ROOT / "anza_ks_k2").glob("*.py")) + [
        ROOT / "scripts/run_anza_ks_k1_5.py",
        ROOT / "tests/test_anza_ks_k1_5.py",
    ]
    sources = [{"path": str(path.relative_to(ROOT)), "sha256": _hash(path)} for path in source_paths]
    protocol = {
        "version": "ANZA_KS_K1_5_FACTORIAL_V1",
        "new_method_only": "K1_E_shear_ks",
        "feature_width": 104,
        "dynamics": "frozen nonhyperbolic shear",
        "information_family": "identical to frozen CatKS K1-D",
        "readout": "StandardScaler plus liblinear LogisticRegression C=1 random_state=17 max_iter=500",
        "splits": "frozen K1 train/dev; confirm hash-only and unopened",
        "gate": "CatKS-ShearKS macro TPR@FPR<=0.05 >=0.04 and paired CI lower >0",
    }
    (RESULT / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    receipt = {
        "status": json.loads((RESULT / "metrics.json").read_text())["status"],
        "protocol_sha256": _hash(RESULT / "protocol.json"),
        "metrics_sha256": _hash(RESULT / "metrics.json"),
        "per_pair_sha256": _hash(RESULT / "per_pair.csv"),
        "report_sha256": _hash(RESULT / "ANZA_KS_K1_5_FACTORIAL_REPORT.md"),
        "source_files": sources,
        "parent_package_sha256": "cd4de1fb01551e616acab9270f984726a8c92264892b2a98559d68001a56df67",
        "old_readouts_retrained": False,
        "confirm_evaluated": False,
        "k2_authorized": True,
    }
    (RESULT / "freeze_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
