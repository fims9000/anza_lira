#!/usr/bin/env python3
"""Build the ANZA-2 CRACKS V2 data and grouped-OOF contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cracks_v2.data_contract import audit_cracks_v2


if __name__ == "__main__":
    result = audit_cracks_v2()
    print(json.dumps({
        "status": result["contract"]["status"],
        "images": result["contract"]["images"]["count"],
        "annotated_images": result["contract"]["pairing"]["image_with_nonexpert_annotation_count"],
        "missing_nominal_ids": result["contract"]["images"]["missing_nominal_1_400"],
        "unannotated_images": result["contract"]["pairing"]["images_without_nonexpert_annotations"],
        "oof_status": result["split"]["status"],
        "expert_data_accessed": False,
    }, indent=2, sort_keys=True))
