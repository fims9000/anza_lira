"""Stable case-local seeds; never dependent on process or wall-clock state."""

from __future__ import annotations

import hashlib

from structural_stability_v1.protocol import FAMILIES, PROTOCOL_ID, SEVERITIES


def perturbation_seed(section_id: int, crop_id: str, family: str, severity: int, view_index: int = 0) -> int:
    if family not in FAMILIES or severity not in SEVERITIES:
        raise ValueError("unknown perturbation family/severity")
    text = "|".join(map(str, (PROTOCOL_ID, int(section_id), crop_id, family, int(severity), int(view_index))))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big", signed=False)

