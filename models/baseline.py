"""Standard Conv2d baseline CNN."""

from __future__ import annotations

import torch.nn as nn


class StandardConvNet(nn.Module):
    """
    Three spatial stages (3×3 Conv + BN + ReLU + MaxPool), then global average pool + linear.

    Widths match the AZConv stages used in this repo so parameter counts are in the same ballpark.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))

