#!/usr/bin/env python3
"""Run the frozen original-ANZA Phase-0 forensic audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from original_anza_forensics.audit import audit_original_anza_operator


if __name__ == "__main__":
    result = audit_original_anza_operator()
    print(json.dumps({
        "status": result["status"],
        "protocol_sha256": result["protocol_sha256"],
        "split_status": result["split_feasibility"]["status"],
        "instrumentation_performed": result["instrumentation_performed"],
        "confirm_performed": result["confirm_performed"],
        "training_performed": result["training_performed"],
        "expert_data_accessed": result["expert_data_accessed"],
        "next_phase_allowed": result["next_phase_allowed"],
    }, indent=2, sort_keys=True))
