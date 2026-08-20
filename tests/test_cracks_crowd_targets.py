from __future__ import annotations

import numpy as np
from PIL import Image

from datasets.cracks import BLUE, GREEN, ORANGE, WHITE, fuse_crowd_masks, map_mask_rgb
from scripts.compute_cracks_normalization import compute_rgb_stats


def _row(*colors: tuple[int, int, int]) -> np.ndarray:
    return np.asarray([colors], dtype=np.uint8)


def test_paper_like_and_conservative_mapping_are_explicit() -> None:
    mask = _row(BLUE, GREEN, ORANGE, WHITE)
    paper_target, paper_valid, paper_confidence = map_mask_rgb(mask, "paper_like")
    conservative_target, conservative_valid, _ = map_mask_rgb(mask, "conservative")
    assert paper_target.tolist() == [[1.0, 1.0, 0.0, 0.0]]
    assert paper_valid.tolist() == [[True, True, False, True]]
    assert paper_confidence.tolist() == [[1.5, 1.0, 1.0, 1.0]]
    assert conservative_target.tolist() == [[1.0, 1.0, 0.0, 0.0]]
    assert conservative_valid.tolist() == [[True, True, True, False]]


def test_practitioner_and_confidence_weights_affect_soft_target() -> None:
    novice = _row(BLUE)
    practitioner = _row(WHITE)
    fused = fuse_crowd_masks([novice, practitioner], ["novice01", "practitioner1"], "paper_like")
    # certain novice positive weight=1*1.5; practitioner background weight=2*1
    assert np.isclose(fused["target"][0, 0], 1.5 / 3.5)
    assert fused["support"][0, 0] == 2
    assert not fused["human_entropy_valid"][0, 0]
    assert np.isfinite(fused["target"]).all()
    assert np.isfinite(fused["human_entropy"]).all()


def test_disagreement_requires_five_valid_annotators() -> None:
    masks = [_row(BLUE), _row(BLUE), _row(WHITE), _row(WHITE), _row(WHITE)]
    names = [f"novice{i:02d}" for i in range(1, 6)]
    fused = fuse_crowd_masks(masks, names, "paper_like")
    assert fused["human_entropy_valid"][0, 0]
    assert 0.0 < fused["human_entropy"][0, 0] <= 1.0
    assert 0.0 <= fused["target"][0, 0] <= 1.0


def test_ignored_pixels_do_not_enter_support_or_target() -> None:
    fused = fuse_crowd_masks([_row(ORANGE), _row(ORANGE)], ["novice01", "practitioner1"], "paper_like")
    assert fused["support"][0, 0] == 0
    assert not fused["valid"][0, 0]
    assert fused["target"][0, 0] == 0.0


def test_rgb_normalization_uses_requested_sections(tmp_path) -> None:
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.full((2, 2, 3), 255, dtype=np.uint8)
    Image.fromarray(first).save(tmp_path / "section_001.png")
    Image.fromarray(second).save(tmp_path / "section_002.png")
    stats = compute_rgb_stats(tmp_path, [1, 2])
    assert stats["section_count"] == 2
    assert np.allclose(stats["mean"], [0.5, 0.5, 0.5])
    assert np.allclose(stats["std"], [0.5, 0.5, 0.5])
