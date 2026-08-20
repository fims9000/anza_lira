from __future__ import annotations

from method_repair.matrix import COMMON_PROTOCOL, protocol_hash, synthetic_matrix


def test_matrix_is_exactly_bounded_a0_to_a4() -> None:
    matrix = synthetic_matrix()
    assert [spec.candidate_id for spec in matrix] == ["A0", "A1", "A2", "A3", "A4"]
    assert len({spec.run_hash for spec in matrix}) == 5
    assert [spec.routing_kernel_size for spec in matrix] == [3, 3, 3, 3, 5]
    assert [spec.use_ambiguity_gate for spec in matrix] == [False, False, True, True, True]
    assert [spec.direct_mode_supervision for spec in matrix] == [False, False, False, True, True]


def test_protocol_forbids_expert_and_old_test_access() -> None:
    assert COMMON_PROTOCOL["expert_access"] == "FORBIDDEN"
    assert COMMON_PROTOCOL["old_test_stream"] == "IMMUTABLE_NOT_USED"
    assert COMMON_PROTOCOL["new_test_stream"].startswith("LOCKED")
    assert len(protocol_hash()) == 16
