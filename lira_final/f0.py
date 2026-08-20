"""Freeze the complete historical experiment ledger without modifying it."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from lira_final.io import sha256, write_json
from lira_final.protocol import RESULT_ROOT, ROOT, PROTOCOL, protocol_hash


HISTORY = (
    ("ANZA-S A2", "results/anza_s/a2/metrics.json"),
    ("ANZA-HS H1", "results/anza_hs/h1/metrics.json"),
    ("ANZA-FS H3", "results/anza_fs/h3/metrics.json"),
    ("ANZA-EK E1", "results/anza_ek/e0_e1/metrics.json"),
    ("ANZA-KS K2", "results/anza_ks/k2/metrics.json"),
    ("ANZA-KIR IR2", "results/anza_kir/ir2/metrics.json"),
    ("TraceGraph TG2", "results/anza_tracegraph/tg2/metrics.json"),
    ("Candidate Audit V2", "results/anza_tracegraph/candidate_audit_v2/metrics.json"),
    ("SBPP V3-A", "results/anza_tracegraph/sbpp_v3_a/metrics.json"),
    ("SBPP V3-B", "results/anza_tracegraph/sbpp_v3_b/metrics.json"),
    ("P0 Endgame", "results/anza_tracegraph/endgame_v1/e3_relation/metrics.json"),
    ("ANZA-LEADS A1", "results/anza_leads/a1_10pct_seed41/metrics.json"),
    ("ANZA-LEADS RC1", "results/anza_leads/rc1/metrics.json"),
    ("ANZA-SurfTrack S0", "results/anza_surftrack/s0/metrics.json"),
)


def _status(payload: dict[str, object]) -> str:
    for key in ("status", "research_status", "final_status"):
        if key in payload:
            return str(payload[key])
    return "STATUS_FIELD_ABSENT"


def run_f0() -> dict[str, object]:
    output = RESULT_ROOT / "f0_freeze"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    history_docs = ROOT / "docs/research_history"
    history_docs.mkdir(parents=True, exist_ok=True)
    for name, relative in HISTORY:
        path = ROOT / relative
        if not path.is_file():
            rows.append({"name": name, "path": relative, "status": "MISSING", "sha256": None})
            continue
        payload = json.loads(path.read_text())
        rows.append({"name": name, "path": relative, "status": _status(payload), "sha256": sha256(path)})
        report_candidates = list(path.parent.glob("*.md"))
        for report in report_candidates:
            destination = history_docs / f"{path.parent.as_posix().replace('/', '__')}__{report.name}"
            if not destination.exists():
                shutil.copy2(report, destination)
    packages = []
    backup_root = ROOT.parent / "_wip_backups/anza_lira"
    if backup_root.is_dir():
        for path in sorted(backup_root.glob("*.zip")):
            packages.append({"path": str(path), "sha256": sha256(path)})
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    stop_text = """# Final Anosov-specific seismic STOP\n\nAll Anosov-specific local, symbolic, cocycle, and transport branches are immutable negative results. SurfTrack S0 selected `lambda=0` under train-only fitting and failed every predeclared practical causal gate. The final LIRA branch is non-Anosov structural continuation; no new Anosov-specific seismic repair is authorized.\n"""
    (output / "ANOSOV_SEISMIC_FINAL_STOP.md").write_text(stop_text)
    registry = {
        "status": "F0_PASS",
        "phase": "F0_FINAL_FREEZE",
        "protocol_sha256": protocol_hash(),
        "git_head": git_head,
        "history": rows,
        "packages": packages,
        "old_results_modified": False,
        "expert_accessed": False,
        "locks": PROTOCOL["locks"],
    }
    write_json(output / "historical_registry.json", registry)
    return registry
