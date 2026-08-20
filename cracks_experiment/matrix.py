"""Frozen Setting A crowd-training matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((PROJECT_ROOT / "results" / "anza_v2_study" / "protocol.json").read_text())
_FROZEN_V2_PATH = PROJECT_ROOT / "results" / "anza_v2_study" / "synthetic" / "frozen_v2.json"
FROZEN_V2 = (
    json.loads(_FROZEN_V2_PATH.read_text())
    if _FROZEN_V2_PATH.exists()
    else {"freeze_sha256": "PENDING_SYNTHETIC_FREEZE", "checkpoint": ""}
)

SETTING_A_PROTOCOL = {
    "setting": "A_crowd_to_expert_same_sections",
    "policy": "paper_like",
    "train_section_count": 393,
    "heldout_annotator_validation_section_count": 392,
    "epochs": 20,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "effective_batch_size": 4,
    "microbatch_size": 1,
    "crop_size": 256,
    "foreground_crop_probability": 0.7,
    "real_loss": "bce+dice+0.2*soft_cldice",
    "topology_iterations": 5,
    "monitoring_validation_crops": 16,
    "final_validation": "all 392 heldout-annotator full sections; tiled overlap 64",
    "threshold_candidates": [round(0.1 + 0.05 * index, 2) for index in range(17)],
    "expert_scores": "LOCKED",
    "cracks_protocol_sha256": PROTOCOL["sha256"],
    "frozen_v2_sha256": FROZEN_V2["freeze_sha256"],
}


@dataclass(frozen=True)
class CRACKSRunSpec:
    run_id: str
    model: str
    seed: int
    structural_replay: bool = False
    use_fuzzy: bool = True
    directional_half_modes: bool = True
    comparison_family: str = "main"

    @property
    def run_hash(self) -> str:
        payload = {"spec": asdict(self), "protocol": SETTING_A_PROTOCOL}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def setting_a_matrix() -> tuple[CRACKSRunSpec, ...]:
    if FROZEN_V2["freeze_sha256"] == "PENDING_SYNTHETIC_FREEZE":
        raise RuntimeError("Setting A remains locked until the synthetic V2 candidate is frozen")
    runs = []
    for seed in (41, 42, 43):
        runs.extend(
            [
                CRACKSRunSpec(f"unet_s{seed}", "unet", seed),
                CRACKSRunSpec(f"deformable_s{seed}", "deformable_unet", seed),
                CRACKSRunSpec(f"v1_s{seed}", "anza_v1", seed),
                CRACKSRunSpec(f"v2_s{seed}", "anza_v2b", seed, structural_replay=True),
            ]
        )
    runs.extend(
        [
            CRACKSRunSpec(
                "v2_no_replay_s42",
                "anza_v2b",
                42,
                structural_replay=False,
                comparison_family="ablation",
            ),
            CRACKSRunSpec(
                "v2_no_fuzzy_s42",
                "anza_v2b",
                42,
                structural_replay=True,
                use_fuzzy=False,
                comparison_family="ablation",
            ),
            CRACKSRunSpec(
                "v2_no_directional_s42",
                "anza_v2a",
                42,
                structural_replay=True,
                directional_half_modes=False,
                comparison_family="ablation",
            ),
        ]
    )
    return tuple(runs)


def setting_a_protocol_hash() -> str:
    encoded = json.dumps(SETTING_A_PROTOCOL, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
