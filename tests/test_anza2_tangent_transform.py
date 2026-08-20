import math

import torch

from models.anza2.losses import transform_doubled_angle


def _axis(theta):
    return torch.tensor([math.cos(2 * theta), math.sin(2 * theta)]).view(1, 1, 2, 1, 1)


def test_t18_rotation_and_flip_transform_axial_targets_exactly() -> None:
    theta = 0.31
    source = _axis(theta)
    torch.testing.assert_close(transform_doubled_angle(source, "rot90"), _axis(theta + math.pi / 2))
    torch.testing.assert_close(transform_doubled_angle(source, "rot180"), _axis(theta), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(transform_doubled_angle(source, "hflip"), _axis(math.pi - theta))
    torch.testing.assert_close(transform_doubled_angle(source, "vflip"), _axis(-theta))
    torch.testing.assert_close(transform_doubled_angle(source, "transpose"), _axis(math.pi / 2 - theta))


def test_t18_transform_moves_spatial_support_with_the_axis() -> None:
    source = torch.zeros(1, 1, 2, 2, 3)
    source[0, 0, :, 0, 2] = _axis(0.0)[0, 0, :, 0, 0]
    rotated = transform_doubled_angle(source, "rot90")
    assert rotated.shape[-2:] == (3, 2)
    torch.testing.assert_close(rotated[0, 0, :, 0, 0], _axis(math.pi / 2)[0, 0, :, 0, 0])
