import numpy as np
import torch

from models.segmentation_v2 import build_comparable_model
from path_completion.anza_guided import (
    anza_guided_conductance,
    filtered_endpoint_pairs,
    image_conductance,
    widest_path_tiebroken,
)
from path_completion.widest_path import EndpointPair


def test_image_and_anza_conductance_are_finite_probabilities():
    image = torch.rand(1, 3, 16, 16)
    model = build_comparable_model("anza_v1", widths=(4, 8, 12, 16))
    generic = image_conductance(image)
    guided = anza_guided_conductance(model.enc1.spatial, image)
    assert generic.shape == guided.shape == (1, 8, 16, 16)
    assert torch.isfinite(guided).all()
    assert torch.all((guided >= 0) & (guided <= 1))


def test_filtered_endpoints_remove_short_spurs():
    mask = np.zeros((32, 32), dtype=bool)
    mask[16, 3:14] = True
    mask[16, 19:30] = True
    mask[5, 5:8] = True
    pairs = filtered_endpoint_pairs(mask, d_min=3, d_max=18, min_branch_length=8)
    assert len(pairs) == 1
    assert pairs[0].first == (16, 13) and pairs[0].second == (16, 19)


def test_tiebroken_widest_path_prefers_stronger_bottleneck():
    relation = np.zeros((8, 7, 7), dtype=np.float32)
    from models.azconv_affinity import LOCAL8_OFFSETS
    right = LOCAL8_OFFSETS.index((1, 0))
    left = LOCAL8_OFFSETS.index((-1, 0))
    for x in range(1, 5):
        relation[right, 3, x] = 0.8
        relation[left, 3, x + 1] = 0.8
    score, path, costs = widest_path_tiebroken(relation, EndpointPair((3, 1), (3, 5), 4.0))
    assert score == np.float32(0.8)
    assert path[0] == (3, 1) and path[-1] == (3, 5)
    assert costs["curvature"] == 0.0
