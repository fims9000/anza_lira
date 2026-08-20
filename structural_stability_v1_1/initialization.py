"""Fresh shared-backbone initialization and historical-checkpoint rejection."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from structural_stability_v1_1.geometry_metric import V11StructuralModel
from structural_stability_v1_1.protocol import PROTOCOL_ID, ROOT, SEEDS, protocol_hash


HISTORICAL_H0_SHA256 = "b2a1115981902620f1b731eaee5a0f4dad6393ae714996726bdaba87dcd3e5f9"


def _seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in state.items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def sidecar_seed(seed: int) -> int:
    text = f"{PROTOCOL_ID}|{int(seed)}|sidecar"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2**31)


def create_shared_backbone_initializations(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in SEEDS:
        _seed_all(seed)
        canonical = V11StructuralModel("B0")
        state = canonical.backbone.state_dict()
        payload = {
            "source": "fresh_v1_1_standard_initialization",
            "protocol_sha256": protocol_hash(),
            "seed": seed,
            "historical_h0_loaded": False,
            "historical_h0_sha256_forbidden": HISTORICAL_H0_SHA256,
            "backbone_state_sha256": state_dict_sha256(state),
            "backbone_state": state,
        }
        path = output / f"backbone_init_s{seed}.pt"
        torch.save(payload, path)
        try:
            display_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            display_path = str(path.resolve())
        records.append({key: value for key, value in payload.items() if key != "backbone_state"} | {"path": display_path})
    return {"status": "FRESH_SHARED_BACKBONE_INITIALIZATIONS_FROZEN", "records": records}


def load_fresh_backbone_initialization(model: V11StructuralModel, path: Path, expected_seed: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("source") != "fresh_v1_1_standard_initialization":
        raise PermissionError("V1.1 refuses non-fresh or historical checkpoint initialization")
    if payload.get("protocol_sha256") != protocol_hash() or payload.get("seed") != int(expected_seed):
        raise ValueError("V1.1 initialization provenance mismatch")
    if payload.get("historical_h0_loaded") is not False:
        raise PermissionError("historical H0 initialization is forbidden")
    state = payload.get("backbone_state")
    if not isinstance(state, dict) or state_dict_sha256(state) != payload.get("backbone_state_sha256"):
        raise ValueError("fresh backbone state hash mismatch")
    model.backbone.load_state_dict(state)
    return {key: value for key, value in payload.items() if key != "backbone_state"}


def initialize_variant(variant: str, seed: int, backbone_path: Path) -> V11StructuralModel:
    if seed not in SEEDS:
        raise ValueError("unplanned V1.1 seed")
    _seed_all(sidecar_seed(seed))
    model = V11StructuralModel(variant)
    load_fresh_backbone_initialization(model, backbone_path, seed)
    return model
