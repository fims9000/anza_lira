from models.anza2.fixtures import fixture_metrics, handcrafted_fixtures


def test_phase1_handcrafted_geometry_fixtures_pass() -> None:
    metrics = fixture_metrics(handcrafted_fixtures())
    assert metrics["phase1_fixture_gate_pass"] is True
    assert metrics["parallel_along_affinity"] > 20 * metrics["parallel_cross_into_gap_affinity"]
    assert metrics["crossing_active_modes"] == 2
    assert metrics["curved_min_path_affinity"] > 0.55
