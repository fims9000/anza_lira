import torch

from affinity_repair.matrix import affinity_matrix, freeze_affinity_protocol
from models.azconv_affinity import StructuralAffinityAZConv2d
from models.segmentation_affinity import build_affinity_model
from models.segmentation_v2 import build_comparable_model


def test_c0_c3_matrix_is_exact_and_no_c4():
    matrix = affinity_matrix()
    assert [spec.candidate_id for spec in matrix] == ["C0", "C1", "C2", "C3"]
    assert len({spec.run_hash for spec in matrix}) == 4
    assert all(spec.seed == 42 for spec in matrix)
    assert matrix[2].affinity and not matrix[2].radius2
    assert matrix[3].affinity and matrix[3].radius2 and matrix[3].hard_ranking


def test_protocol_freeze_refuses_drift(tmp_path):
    path = tmp_path / "protocol.json"
    freeze_affinity_protocol(path)
    freeze_affinity_protocol(path)
    path.write_text("{}\n")
    try:
        freeze_affinity_protocol(path)
    except ValueError as error:
        assert "drift" in str(error)
    else:
        raise AssertionError("protocol drift was accepted")


def test_models_are_seed_matched_and_affinity_is_direct_operator():
    torch.manual_seed(42)
    v1 = build_comparable_model("anza_v1", widths=(4, 6, 8, 10))
    c1 = build_affinity_model("C1", widths=(4, 6, 8, 10), seed_matched_v1=v1)
    c3 = build_affinity_model("C3", widths=(4, 6, 8, 10), seed_matched_v1=v1)
    assert isinstance(c3.enc1.spatial, StructuralAffinityAZConv2d)
    torch.testing.assert_close(c1.enc2.spatial.pointwise.weight, v1.enc2.spatial.pointwise.weight)
    output = c3(torch.randn(1, 3, 32, 32), return_diagnostics=True)
    assert output["visible_logits"].shape == (1, 1, 32, 32)
    assert output["affinity_diagnostics"]["radius2_affinity"] is not None
