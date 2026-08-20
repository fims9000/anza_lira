import numpy as np
import pytest
import torch

from cracks_experiment.validation import _tile_starts, freeze_setting_a_thresholds, select_threshold, tiled_probability


class PointModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :1] * self.scale


def test_tiling_covers_final_edge_and_blends_pointwise_model_exactly() -> None:
    assert _tile_starts(704, 256, 64) == (0, 192, 384, 448)
    image = torch.linspace(-2, 2, 256 * 704).reshape(1, 256, 704).repeat(3, 1, 1)
    actual = tiled_probability(PointModel(), image)
    expected = torch.sigmoid(image[0])
    assert actual.shape == (256, 704)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_threshold_selection_is_macro_section_dice_and_ties_choose_lower() -> None:
    rows = []
    for threshold, dice_values in ((0.4, (1.0, 0.5)), (0.5, (0.75, 0.75))):
        for section, dice in enumerate(dice_values):
            rows.append(
                {
                    "section_id": section,
                    "threshold": threshold,
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                    "tn": 1,
                    "dice": dice,
                    "iou": dice,
                }
            )
    selected = select_threshold(rows)
    assert selected["selected_threshold"] == pytest.approx(0.4)
    assert selected["selection_metric"] == "macro_section_dice"
    assert np.isfinite([row["micro_dice"] for row in selected["sweep"]]).all()


def test_threshold_freeze_fails_closed_when_matrix_is_incomplete(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Crowd validation missing"):
        freeze_setting_a_thresholds(tmp_path)
