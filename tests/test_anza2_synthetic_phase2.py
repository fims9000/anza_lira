from anza2_experiment.synthetic_mechanism import (
    METHODS,
    _branch_fixture,
    _path_fixture,
    _relations,
    protocol_payload,
)
from structural.widest_path import domain_restricted_widest_path
from models.anza2.affinity import LOCAL8_OFFSETS


def test_phase2_protocol_is_zero_train_and_expert_locked() -> None:
    protocol = protocol_payload()
    assert protocol["training_performed"] is False
    assert protocol["cracks_data_accessed"] is False
    assert protocol["expert_data_accessed"] is False
    assert protocol["target_fpr"] == 0.05


def test_parallel_confuser_is_rejected_by_displacement_aware_anza_relation() -> None:
    fixture = _path_fixture("parallel_false_bridge", 610_020_001)
    relations = _relations(fixture)
    scores = {}
    for method in METHODS:
        scores[method], _path = domain_restricted_widest_path(
            relations[method], fixture["start"], fixture["goal"],
            domain=fixture["domain"], offsets=LOCAL8_OFFSETS,
        )
    assert scores["anza2_absolute"] < 0.05
    assert scores["simple_axis_similarity"] > scores["anza2_absolute"] * 5


def test_crossing_oracle_field_supports_every_incident_branch() -> None:
    fixture = _branch_fixture("x_crossing", 610_100_001)
    relation = _relations(fixture)["anza2_absolute"]
    scores = [relation[channel, point[0], point[1]] for channel, point in fixture["expected_edges"]]
    assert len(scores) == 4
    assert min(scores) > 0.6
