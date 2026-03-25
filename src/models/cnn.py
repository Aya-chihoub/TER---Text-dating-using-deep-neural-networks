"""
Multi-branch 1-D CNN for author-age prediction.

Architecture — 3 parallel branches, each with 3 stacked (Conv1d → MaxPool) layers:

    Input: (batch, seq_len, feature_dim) → transpose → (batch, feature_dim, seq_len)

    Branch k=3:   [Conv1d(k=3) → BN → ReLU → MaxPool(2)] × 3
    Branch k=7:   [Conv1d(k=7) → BN → ReLU → MaxPool(2)] × 3
    Branch k=13:  [Conv1d(k=13) → BN → ReLU → MaxPool(2)] × 3

    Global Average Pooling on each branch → Concatenation → Dropout → Linear → num_classes
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.config import ModelConfig


class _Branch(nn.Module):
    """One parallel branch: 3 stacked Conv1d + MaxPool layers with the same kernel size."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, pool_size: int, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            c_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv1d(c_in, out_channels, kernel_size=kernel_size, padding=0))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(kernel_size=pool_size))
        self.layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.global_pool(x).squeeze(-1)      # (batch, out_channels)


class AgeCNN(nn.Module):
    """
    Multi-branch temporal 1-D CNN that classifies a sequence of
    hand-crafted word-level features into one of *num_classes* age buckets.

    Each branch uses a different kernel size (3, 7, 13) and stacks
    3 Conv1d + MaxPool(2) layers.  The outputs are concatenated
    and passed through a linear classifier.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()

        self.branches = nn.ModuleList()
        for k in cfg.kernel_sizes:
            branch = _Branch(
                in_channels=cfg.feature_dim,
                out_channels=cfg.num_filters,
                kernel_size=k,
                pool_size=cfg.pool_size,
                num_layers=cfg.num_conv_layers,
            )
            self.branches.append(branch)

        total_filters = cfg.num_filters * len(cfg.kernel_sizes)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(total_filters, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, feature_dim)

        Returns
        -------
        logits : Tensor of shape (batch, num_classes)
        """
        x = x.transpose(1, 2)                       # (batch, feature_dim, seq_len)
        branch_outs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outs, dim=1)            # (batch, total_filters)
        x = self.dropout(x)
        return self.fc(x)
