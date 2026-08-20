"""Data/access/initialization contracts for V1.1 SS1.5."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from structural_stability_v1.perturb.seeds import perturbation_seed
from structural_stability_v1_1.geometry_metric import V11StructuralModel
from structural_stability_v1_1.initialization import (
    create_shared_backbone_initializations,
    initialize_variant,
    load_fresh_backbone_initialization,
    state_dict_sha256,
)
from structural_stability_v1_1.protocol import PROTOCOL, SEEDS, VARIANTS
from structural_stability_v1_1.train_variants import assert_training_inputs
from structural_stability_v1_1.training_manifest import assert_manifest_shared, selected_crop_has_explicit


def test_v11_freezes_three_seeds_four_variants_and_36_epochs() -> None:
    assert SEEDS == (41, 42, 43)
    assert VARIANTS == ("B0", "B1", "B2", "B3")
    assert PROTOCOL["planned_training_jobs"] == 12
    assert PROTOCOL["training"]["epochs"] == 36
    assert PROTOCOL["training"]["planned_optimizer_updates"] == 1980
    assert PROTOCOL["training"]["training_severities"] == [1, 2]
    assert PROTOCOL["training"]["evaluation_only_severity"] == 3


def test_parent_perturbation_seed_contract_is_unchanged() -> None:
    assert perturbation_seed(235, "full_section", "warp", 3, 0) == perturbation_seed(235, "full_section", "warp", 3, 0)
    assert perturbation_seed(235, "full_section", "warp", 3, 0) != perturbation_seed(235, "full_section", "warp", 3, 1)


def test_manifest_must_be_shared_and_exclude_severity_three() -> None:
    assert_manifest_shared({"consumers": list(VARIANTS), "severities": [1, 2]})
    with pytest.raises(ValueError):
        assert_manifest_shared({"consumers": ["B3"], "severities": [1, 2]})
    with pytest.raises(ValueError):
        assert_manifest_shared({"consumers": list(VARIANTS), "severities": [1, 2, 3]})


def test_selected_crop_explicit_guard_is_fail_closed() -> None:
    # Prefix arrays include the leading zero and describe a padded 704-wide row.
    prefix = torch.zeros(705, dtype=torch.int64).numpy()
    prefix[101:] = 3
    assert selected_crop_has_explicit({"a": prefix}, ["a"], 0, 256)
    assert not selected_crop_has_explicit({"a": prefix}, ["a"], 300, 256)


def test_training_access_guard_rejects_nontrain_or_evaluation_access() -> None:
    assert_training_inputs(section_ids=list(range(220)), expert_accessed=False, development_accessed=False, confirm_accessed=False)
    with pytest.raises(PermissionError):
        assert_training_inputs(section_ids=list(range(219)), expert_accessed=False, development_accessed=False, confirm_accessed=False)
    with pytest.raises(PermissionError):
        assert_training_inputs(section_ids=list(range(220)), expert_accessed=True, development_accessed=False, confirm_accessed=False)


def test_fresh_backbone_state_is_identical_across_variants(tmp_path: Path) -> None:
    manifest = create_shared_backbone_initializations(tmp_path)
    record = next(row for row in manifest["records"] if row["seed"] == 41)
    hashes = []
    for variant in VARIANTS:
        model = initialize_variant(variant, 41, tmp_path / "backbone_init_s41.pt")
        hashes.append(state_dict_sha256(model.backbone.state_dict()))
    assert len(set(hashes)) == 1 == len({record["backbone_state_sha256"]})


def test_historical_or_unproven_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "historical.pt"
    torch.save({"status": "COMPLETE", "model_state": V11StructuralModel("B0").backbone.state_dict()}, path)
    with pytest.raises(PermissionError):
        load_fresh_backbone_initialization(V11StructuralModel("B0"), path, 41)


def test_model_sidecars_do_not_change_segmentation_interface() -> None:
    image = torch.randn(1, 3, 64, 64)
    for variant in VARIANTS:
        model = V11StructuralModel(variant).eval()
        with torch.no_grad():
            plain = model(image)
            diagnostics = model(image, return_geometry=True)
        assert plain.shape == (1, 1, 64, 64)
        torch.testing.assert_close(plain, diagnostics["visible_logits"])
        assert len(diagnostics["geometry"]) == (2 if variant in {"B2", "B3"} else 0)
