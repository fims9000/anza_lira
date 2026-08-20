import hashlib
from pathlib import Path

import torch

from models.azconv import AZConv2d, AZConvConfig
from original_anza_forensics.audit import (
    LEGACY_SOURCE_SHA256,
    _split_feasibility,
    inspect_legacy_layer,
)


def test_read_only_reconstruction_exactly_matches_legacy_forward() -> None:
    torch.manual_seed(5)
    layer = AZConv2d(3, 4, num_rules=3, cfg=AZConvConfig()).eval()
    result = inspect_legacy_layer(layer, torch.randn(1, 3, 7, 9))
    assert result["forward_reconstruction_max_abs_error"] < 1e-6
    assert result["membership_sum_max_abs_error"] < 1e-6
    assert result["normalization_sum_max_abs_error"] < 1e-6
    assert result["all_finite"] is True
    assert result["tensor_shapes"]["w_raw_per_mode"] == [1, 3, 9, 63]
    assert result["tensor_shapes"]["W_raw_mode_sum"] == [1, 9, 63]


def test_legacy_source_is_unchanged_and_uses_softmax() -> None:
    path = Path("models/azconv.py")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == LEGACY_SOURCE_SHA256
    source = path.read_text()
    assert "mu = F.softmax" in source
    assert "compat = mu_center * mu_un * kern * valid_un" in source
    assert "compat.sum(dim=(1, 2)" in source


def test_no_independent_annotated_confirm_split_exists() -> None:
    split = _split_feasibility()
    assert split["unseen_image_section_ids"] == [49, 73, 385]
    assert split["unseen_annotation_counts"] == {"49": 0, "73": 0, "385": 0}
    assert split["eligible_independent_nonexpert_confirm_section_ids"] == []
    assert split["status"] == "STOP_NO_INDEPENDENT_CONFIRM_SPLIT"
