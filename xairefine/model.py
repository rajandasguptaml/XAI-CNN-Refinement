"""Baseline CNN of Sec. 3.3, built from a list of block widths so that blocks can be removed and the network rebuilt.

Each block is Conv(3x3, same padding) + ReLU + MaxPool(2). After the blocks: batch normalisation, global average
pooling, flatten, two fully connected layers of 512 units with ReLU, and a 4-way output (softmax is applied by the
loss). The baseline widths are 8, 16, 32, 64, 128, 256.
"""
from __future__ import annotations

import torch
from torch import nn

BASELINE_FILTERS = [8, 16, 32, 64, 128, 256]


class Block(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.act = nn.ReLU(inplace=False)  # its output is the Grad-CAM target of this block
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.act(self.conv(x)))


class XCNN(nn.Module):
    def __init__(self, filters: list[int] | None = None, num_classes: int = 4, in_channels: int = 3, hidden: int = 512):
        super().__init__()
        self.filters = list(filters or BASELINE_FILTERS)
        if not self.filters:
            raise ValueError("at least one convolutional block is required")
        chans = [in_channels] + self.filters
        self.blocks = nn.ModuleList(Block(a, b) for a, b in zip(chans[:-1], chans[1:]))
        self.bn = nn.BatchNorm2d(self.filters[-1])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.filters[-1], hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return self.head(self.gap(self.bn(x)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def to_input(batch_uint8: torch.Tensor) -> torch.Tensor:
    """(B, H, W) uint8 grayscale -> (B, 3, H, W) float in [0, 1]."""
    x = batch_uint8.float().div_(255.0).unsqueeze(1)
    return x.expand(-1, 3, -1, -1).contiguous()
