"""
DINOv3 Structural Teacher — Relation Distillation.

Freezes DINOv3 ViT-L/16 and provides per-layer spatial feature maps.
Used for Structural Relation Distillation: CLIP learns to reproduce
DINOv3's patch-patch similarity structure without feature-space alignment.

Architecture:
    DINOv3 ViT-L/16 (frozen) → per-layer feature maps [B, C, H, W]
    → adaptive pool to grid → similarity matrix → L_struct target
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOTeacher(nn.Module):
    """
    Frozen DINOv3 ViT-L/16 backbone for structural relation distillation.

    Provides per-layer spatial feature maps for computing patch similarity
    matrices. CLIP's patch features are trained to match DINOv3's similarity
    structure (not absolute feature values).

    Args:
        layers: list of layer indices to extract (default [6, 12, 18, 24])
        device: target device
    """

    def __init__(self, layers=(6, 12, 18, 24), device='cuda:0'):
        super().__init__()
        self.layers = layers
        self.device = device

        out_indices = tuple(l - 1 for l in layers)
        self.backbone = timm.create_model(
            'vit_large_patch16_dinov3',
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
        ).to(device).eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = 1024

    def _preprocess(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        if x.dtype != next(self.backbone.parameters()).dtype:
            x = x.to(next(self.backbone.parameters()).dtype)
        _, _, H, W = x.shape
        H16 = (H // 16) * 16
        W16 = (W // 16) * 16
        if H != H16 or W != W16:
            x = F.interpolate(x, size=(H16, W16), mode='bilinear', align_corners=False)
        return x

    @torch.no_grad()
    def get_feature_maps(self, x):
        """
        Extract per-layer spatial feature maps.

        Args:
            x: [B, 3, H, W]
        Returns:
            list of [B, 1024, h, w] per layer
        """
        x = self._preprocess(x)
        return self.backbone(x)
