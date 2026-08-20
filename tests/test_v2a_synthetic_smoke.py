from __future__ import annotations

import torch
import torch.nn.functional as F

from models.azconv_v2 import AZConvV2Config, ModeResolvedAZConv2d
from synthetic.crossing_trace_bench import CrossingTraceBench


def test_v2a_runs_one_finite_visible_segmentation_step_on_synthetic_train_stream() -> None:
    torch.manual_seed(43)
    dataset = CrossingTraceBench("train", image_size=32, length=2)
    images = torch.stack([dataset[index]["image"] for index in range(2)])
    visible_targets = torch.stack([dataset[index]["visible_fault_mask"] for index in range(2)])
    layer = ModeResolvedAZConv2d(
        3,
        1,
        cfg=AZConvV2Config(num_modes=4, state_channels=4, transport_steps=1),
    )
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
    diagnostics = layer(images, return_diagnostics=True)
    loss = F.binary_cross_entropy_with_logits(diagnostics["output"], visible_targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert diagnostics["output"].shape == visible_targets.shape
    assert diagnostics["mode_states"].shape[:2] == (2, 4)
    assert torch.isfinite(diagnostics["mode_states"]).all()


def test_v2b_runs_one_finite_visible_segmentation_step_on_synthetic_train_stream() -> None:
    torch.manual_seed(46)
    dataset = CrossingTraceBench("train", image_size=16, length=1)
    image = dataset[0]["image"].unsqueeze(0)
    visible_target = dataset[0]["visible_fault_mask"].unsqueeze(0)
    layer = ModeResolvedAZConv2d(
        3,
        1,
        cfg=AZConvV2Config(
            num_modes=3,
            state_channels=3,
            transport_steps=1,
            variant="v2b",
        ),
    )
    diagnostics = layer(image, return_diagnostics=True)
    loss = F.binary_cross_entropy_with_logits(diagnostics["output"], visible_target)
    loss.backward()
    assert torch.isfinite(loss)
    assert diagnostics["mode_states"].shape == (1, 3, 2, 3, 16, 16)
    assert torch.isfinite(diagnostics["transport_mass"]).all()
