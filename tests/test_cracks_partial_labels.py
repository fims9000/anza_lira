import numpy as np
from PIL import Image
import pytest
import torch

from cracks_experiment.partial_labels import (
    CRACKSMultiAnnotatorDataset,
    audit_nonexpert_annotations,
    average_annotator_loss,
    map_partial_annotation,
)
from datasets.cracks import BLUE, GREEN, ORANGE, WHITE


def test_partial_label_color_contract_and_unknown_failure() -> None:
    mask = np.asarray([[BLUE, GREEN, ORANGE, WHITE]], dtype=np.uint8)
    target, weight = map_partial_annotation(mask)
    assert target.tolist() == [[1.0, 1.0, 0.0, 0.0]]
    assert weight.tolist() == [[1.0, 0.5, 1.0, 0.0]]
    mask[0, 0] = (1, 2, 3)
    with pytest.raises(ValueError, match="outside frozen T1 semantics"):
        map_partial_annotation(mask)


def test_white_pixels_have_exactly_zero_gradient() -> None:
    logits = torch.zeros((1, 1, 16, 16), requires_grad=True)
    targets = torch.zeros((1, 1, 16, 16))
    weights = torch.zeros_like(targets)
    weights[0, 0, 8, 8] = 1.0
    targets[0, 0, 8, 8] = 1.0
    loss, _ = average_annotator_loss(logits, targets, weights, topology_weight=0.0)
    loss.backward()
    gradient = logits.grad.detach()
    assert gradient[0, 0, 8, 8] != 0
    assert torch.count_nonzero(gradient) == 1


def test_annotator_losses_are_averaged_not_masks_fused() -> None:
    logits = torch.tensor([[[[2.0, -2.0]]]])
    targets = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    weights = torch.ones_like(targets)
    combined, logs = average_annotator_loss(logits, targets, weights, topology_weight=0.0)
    first, _ = average_annotator_loss(logits, targets[:1], weights[:1], topology_weight=0.0)
    second, _ = average_annotator_loss(logits, targets[1:], weights[1:], topology_weight=0.0)
    assert torch.allclose(combined, (first + second) / 2)
    assert logs["annotator_count"] == 2.0


def _write_section(root, annotator: str, color) -> None:
    directory = root / annotator
    directory.mkdir(parents=True, exist_ok=True)
    mask = np.full((255, 701, 3), WHITE, dtype=np.uint8)
    mask[120:123, 330:370] = color
    Image.fromarray(mask).save(directory / "section_001.png")


def test_dataset_keeps_annotations_separate_and_rejects_expert(tmp_path) -> None:
    image_root = tmp_path / "images"
    annotation_root = tmp_path / "annotations"
    image_root.mkdir()
    Image.fromarray(np.zeros((255, 701, 3), dtype=np.uint8)).save(image_root / "section_001.png")
    _write_section(annotation_root, "novice01", BLUE)
    _write_section(annotation_root, "practitioner1", GREEN)
    dataset = CRACKSMultiAnnotatorDataset(
        image_root,
        annotation_root,
        [1],
        ["novice01", "practitioner1"],
        mean=(0, 0, 0),
        std=(1, 1, 1),
        crop_size=256,
        foreground_probability=1.0,
        annotators_per_section=None,
        seed=9,
    )
    item = dataset[0]
    assert item["targets"].shape == (2, 1, 256, 256)
    assert item["weights"][0].max() == 1.0
    assert item["weights"][1].max() == 0.5
    assert item["annotators"] == ("novice01", "practitioner1")
    with pytest.raises(PermissionError, match="forbids expert"):
        CRACKSMultiAnnotatorDataset(
            image_root,
            annotation_root,
            [1],
            ["expert"],
            mean=(0, 0, 0),
            std=(1, 1, 1),
        )


def test_palette_audit_never_accepts_expert(tmp_path) -> None:
    _write_section(tmp_path, "novice01", ORANGE)
    result = audit_nonexpert_annotations(tmp_path, ["novice01"], [1])
    assert result["status"] == "PASS"
    assert result["palette"]["pixels"]["orange"] == 120
    assert result["expert_data_accessed"] is False
    with pytest.raises(PermissionError):
        audit_nonexpert_annotations(tmp_path, ["expert"], [1])
