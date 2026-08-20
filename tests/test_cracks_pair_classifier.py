import numpy as np
import torch

from path_completion.cracks_pair_training import CRACKSRealPairClassifier, _operating_point
from path_completion.cracks_pairs import (
    matched_section_pairs,
    oriented_real_pair_crop,
    split_sections,
)


def _toy() -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((96, 128), dtype=bool)
    mask[35, 5:45] = True
    mask[35, 55:110] = True
    image = np.zeros((3, 96, 128), dtype=np.float32)
    image[:, 34:37, mask[35]] = 0.8
    return mask, image


def test_real_pair_split_is_deterministic_and_section_disjoint() -> None:
    first = split_sections(range(1, 101))
    second = split_sections(range(1, 101))
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == set(range(1, 101))


def test_positive_and_negative_pairs_have_distinct_lineage() -> None:
    mask, image = _toy()
    pairs = matched_section_pairs(mask, image, max_pairs=2)
    assert pairs
    positive, negative = pairs[0]
    assert positive.label == 1
    assert positive.source_kind == "same_trace_internal_gap"
    assert negative.label == 0
    assert negative.source_kind == "different_connected_traces"
    assert 6 <= negative.distance <= 24
    assert np.all(np.abs(positive.descriptor - negative.descriptor) <= np.asarray([6.0, 0.35, 0.08, 0.15]))


def test_real_pair_crop_is_deterministic_eight_channel_and_erases_bridge() -> None:
    mask, image = _toy()
    positive, _ = matched_section_pairs(mask, image, max_pairs=1)[0]
    fields = {
        "image": image,
        "base_probability": mask.astype(np.float32),
        "cos2theta": np.ones(mask.shape, dtype=np.float32),
        "sin2theta": np.zeros(mask.shape, dtype=np.float32),
        "anisotropy": np.full(mask.shape, 0.5, dtype=np.float32),
    }
    first = oriented_real_pair_crop(fields, positive)
    second = oriented_real_pair_crop(fields, positive)
    assert first.shape == (8, 33, 49)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert first[4].max() == 1.0
    assert first[7].mean() == np.float32(0.5)
    assert first[3, 16, 24] < 0.1


def test_real_pair_classifier_shape_and_operating_point_gate() -> None:
    model = CRACKSRealPairClassifier()
    assert model(torch.zeros((3, 8, 33, 49))).shape == (3,)
    positive = np.asarray([0.95, 0.9, 0.85, 0.8])
    negative = np.asarray([0.1, 0.2, 0.3, 0.4])
    operating = _operating_point(positive, negative)
    assert operating["fpr"] <= 0.05
    assert operating["threshold"] == 0.8
    assert operating["tpr"] == 1.0
