"""
Score fusion utilities: z-score based fusion of topk and mutual scores.
"""

import torch


def fuse_scores(topk_scores, mutual_scores, weight=0.5):
    """
    Fuse top-k mean scores with mutual scores via z-score normalization.

    Each component is independently z-scored to remove scale differences,
    then averaged with the given weight.

    Args:
        topk_scores: [B] image-level scores from anomaly map aggregation
        mutual_scores: [B] image-level scores from mutual rarity
        weight: weight for topk_scores (0 = pure mutual, 1 = pure topk, default 0.5)

    Returns:
        fused: [B] combined scores
    """
    def zscore(x):
        s = x.std()
        if s < 1e-8:
            return torch.zeros_like(x)
        return (x - x.mean()) / s

    z_topk = zscore(topk_scores)
    z_mutual = zscore(mutual_scores)
    return weight * z_topk + (1.0 - weight) * z_mutual
