import torch

from anza2.forensics.component_replacement import (
    align_learned_field, component_replacements, oracle_field_from_sample,
)
from anza2_experiment.learned_affinity import LearnedAffinityModel
from models.anza2.affinity import ANZA2StructuralAffinity
from synthetic.crossing_trace_bench_v4 import generate_sample_v4


def test_f0_f1_and_duplicate_f2_f9_are_exact():
    sample = generate_sample_v4("validation", 256, image_size=64)
    model = LearnedAffinityModel().eval()
    with torch.inference_mode():
        output = model(torch.as_tensor(sample["image"]).unsqueeze(0), use_anza=True)
    oracle, valid = oracle_field_from_sample(sample, device=torch.device("cpu"))
    learned, _ = align_learned_field(output["field"], oracle, valid)
    matrix = component_replacements(oracle, learned)
    assert torch.equal(matrix["F0_full_oracle"].membership, oracle.membership)
    assert torch.equal(matrix["F1_full_learned"].orientation, learned.orientation)
    assert torch.equal(matrix["F2_learned_membership_only"].membership, matrix["F9_learned_membership_oracle_geometry"].membership)
    assert torch.equal(matrix["F2_learned_membership_only"].orientation, matrix["F9_learned_membership_oracle_geometry"].orientation)


def test_component_replacement_retains_mode_permutation_invariant_affinity():
    sample = generate_sample_v4("validation", 256, image_size=64)
    oracle, _ = oracle_field_from_sample(sample, device=torch.device("cpu"))
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = type(oracle)(*(getattr(oracle, name)[:, permutation] for name in (
        "membership", "orientation", "base_scale", "hyperbolicity", "sigma_parallel", "sigma_perpendicular"
    )))
    affinity = ANZA2StructuralAffinity()
    assert torch.allclose(affinity(oracle), affinity(permuted), atol=1e-7)
