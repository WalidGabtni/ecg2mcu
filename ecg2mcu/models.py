"""The 1D-CNN that is exported to the microcontroller.

Four conv blocks (32, 64, 128, 256 filters; kernels 7, 5, 3, 3), global average pooling and
a two-layer classifier. Layer names match the thesis checkpoints, so those load unchanged.
Only plain layers are used: they map one-to-one to TFLite Micro INT8 operators.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import CLASS_NAMES

CONV_SPECS = [(32, 7), (64, 5), (128, 3), (256, 3)]  # (filters, kernel size)


class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = len(CLASS_NAMES),
                 fc_size: int = 256, dropout: float = 0.3):
        super().__init__()
        self.in_channels = in_channels
        self.fc_size = fc_size

        prev = in_channels
        for i, (filters, kernel) in enumerate(CONV_SPECS, start=1):
            setattr(self, f"conv{i}", nn.Conv1d(prev, filters, kernel_size=kernel, padding=kernel // 2, bias=False))
            setattr(self, f"bn{i}", nn.BatchNorm1d(filters))
            setattr(self, f"relu{i}", nn.ReLU(inplace=True))
            if i < len(CONV_SPECS):
                setattr(self, f"pool{i}", nn.MaxPool1d(2))
            prev = filters

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(prev, fc_size)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(1, len(CONV_SPECS) + 1):
            x = getattr(self, f"relu{i}")(getattr(self, f"bn{i}")(getattr(self, f"conv{i}")(x)))
            if i < len(CONV_SPECS):
                x = getattr(self, f"pool{i}")(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(self.relu_fc(self.fc1(x)))
        return self.fc2(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
