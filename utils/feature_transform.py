"""
Feature transformation modules: MLP, residual MLP, and shared bottleneck adapter.
"""

import torch
import torch.nn as nn


class FeatureTransformMLP(nn.Module):
    """Complete MLP module for more complex feature transformations"""
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        if output_dim is None:
            output_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class ResidualMLPTransform(nn.Module):
    """Standard MLP with residual branch to both enhance expressiveness and preserve CLIP original features"""

    def __init__(self, input_dim, hidden_dim=None, output_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(16, input_dim // 2)
        if output_dim is None:
            output_dim = input_dim

        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

        self.residual_proj = None
        if output_dim != input_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x if self.residual_proj is None else self.residual_proj(x)
        x = self.norm(x)
        x = self.mlp(x)
        return residual + x


class SharedBottleneckAdapter(nn.Module):
    """Lightweight shared bottleneck adapter: 1024 → 64 → 1024 with per-layer scales.

    Replaces 4 independent 2M-param MLPs (8.4M total) with shared projections
    (131K params, 64× reduction). The bottleneck forces the adapter to learn
    essential transformations, reducing overfitting on small datasets like VisA.

    Args:
        embed_dim: Feature dimension (1024 for ViT-L)
        bottleneck: Bottleneck dimension (default 64)
        num_layers: Number of feature layers (default 4)
        init_scale: Initial value for per-layer residual scales
    """

    def __init__(self, embed_dim=1024, bottleneck=64, num_layers=4, init_scale=0.01):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck = bottleneck
        self.num_layers = num_layers

        # Shared down/up projections
        self.down = nn.Linear(embed_dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, embed_dim, bias=False)

        # Per-layer residual scale (learnable)
        self.layer_scales = nn.Parameter(torch.ones(num_layers) * init_scale)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)  # Zero-init up → adapter starts as identity

    def forward(self, x, layer_idx):
        """
        Args:
            x: [N, embed_dim] flattened patch features
            layer_idx: 0-based index into layer_scales
        Returns:
            [N, embed_dim] adapted features
        """
        adapted = self.up(self.act(self.down(x)))
        scale = self.layer_scales[layer_idx]
        return x + scale * adapted
