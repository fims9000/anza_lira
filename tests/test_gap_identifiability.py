from __future__ import annotations

from affinity_repair.identifiability import IDENTIFIABILITY_PROTOCOL, pair_distance_row


def test_matched_pairs_are_pixel_observable_not_contradictory_identical_inputs() -> None:
    for pair_id in range(8):
        row = pair_distance_row(pair_id, image_size=64)
        assert row["exact_pixel_equal"] is False
        assert row["mean_absolute_difference"] > IDENTIFIABILITY_PROTOCOL["practical_identity_mae_maximum"]
        assert row["endpoint_patch_mae"] > 0.0


def test_matching_metadata_is_exact_while_pixels_remain_different() -> None:
    row = pair_distance_row(17, image_size=96)
    assert row["geometry_seed_equal"] is True
    assert row["render_difficulty_seed_equal"] is True
    assert row["gap_length_difference_px"] == 0.0
    assert row["endpoint_distance_difference_px"] == 0.0
    assert row["local_orientation_difference_rad"] == 0.0
    assert row["phase_metadata_status"] == "NOT_STORED"
