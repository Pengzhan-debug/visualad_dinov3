"""
Spatial Context Refinement for anomaly maps.

A lightweight learnable module that refines per-pixel anomaly scores using
neighborhood context. Unlike a fixed Gaussian filter, this learns to:
  - Boost isolated high scores (true defects) surrounded by normal regions
  - Suppress scattered high scores (texture noise) that lack spatial coherence

Design:
  - 3×3 depthwise convolution: 1 in → 1 out channel
  - Initialized as identity (center=1.0, rest=0.0) so it's a no-op at start
  - Residual connection with learnable alpha for stable training
  - Only 10 parameters total

Usage:
    from utils.spatial_refinement import SpatialRefinement
    refiner = SpatialRefinement()
    refined_map = refiner(anomaly_map)  # [B, H, W] or [H, W]
"""

import torch
import torch.nn as nn


class SpatialRefinement(nn.Module):
    """
    Learnable spatial context refinement for anomaly maps.

    Args:
        init_alpha: Initial value for the residual scale parameter.
                    Default 0.1 means the module starts as near-identity.
    """

    def __init__(self, init_alpha: float = 0.1):
        super().__init__()
        # 3×3 conv, 1 channel in, 1 channel out, no bias
        self.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)

        # Initialize as identity: center weight = 1.0, others = 0.0
        nn.init.constant_(self.conv.weight, 0.0)
        self.conv.weight.data[0, 0, 1, 1] = 1.0

        # Learnable residual scale — starts small to preserve original signal
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anomaly_map: [H, W] or [B, H, W] anomaly map
        Returns:
            refined: Same shape as input, with spatial context applied
        """
        needs_squeeze = anomaly_map.dim() == 2
        if needs_squeeze:
            anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif anomaly_map.dim() == 3:
            anomaly_map = anomaly_map.unsqueeze(1)  # [B, 1, H, W]

        # 3×3 spatial context aggregation
        context = self.conv(anomaly_map)  # [B, 1, H, W]

        # Residual: anomaly_map + alpha * (context - anomaly_map)
        # At init: context ≈ anomaly_map (identity conv), so output ≈ anomaly_map
        # After training: alpha controls how much spatial context matters
        refined = anomaly_map + self.alpha * (context - anomaly_map)

        refined = refined.squeeze(1)  # [B, H, W]
        if needs_squeeze:
            refined = refined.squeeze(0)  # [H, W]

        return refined


__all__ = ["SpatialRefinement"]
