from __future__ import annotations

from synthetic.experiment_matrix import COMMON_PROTOCOL, development_matrix, protocol_hash


def test_architecture_only_runs_share_the_exact_visible_objective() -> None:
    runs = development_matrix()
    architecture_only = [run for run in runs if run.comparison_family == "architecture_only"]
    assert [run.model for run in architecture_only] == [
        "unet",
        "deformable_unet",
        "anza_v1",
        "anza_v2a",
        "anza_v2b",
    ]
    assert {run.objectives for run in architecture_only} == {("visible_bce_dice",)}
    assert COMMON_PROTOCOL["visible_segmentation_loss"] == "bce_plus_dice"


def test_structural_supervision_is_a_separate_predeclared_line() -> None:
    by_id = {run.candidate_id: run for run in development_matrix()}
    assert by_id["C2"].model == by_id["C3"].model == "anza_v2b"
    assert by_id["C2"].objectives == ("visible_bce_dice",)
    assert by_id["C3"].objectives == ("visible_bce_dice", "route")
    assert by_id["C4"].objectives == (
        "visible_bce_dice",
        "route",
        "positive_negative_gap",
    )
    assert by_id["C5"].objectives[-1] == "cone"


def test_development_budget_and_hashes_are_frozen() -> None:
    candidates = [run for run in development_matrix() if run.candidate_id.startswith("C")]
    assert [run.candidate_id for run in candidates] == [f"C{index}" for index in range(6)]
    hashes = [run.run_hash for run in development_matrix()]
    assert len(hashes) == len(set(hashes))
    assert len(protocol_hash()) == 16
    assert COMMON_PROTOCOL["test_stream"] == "FROZEN_UNOPENED"
