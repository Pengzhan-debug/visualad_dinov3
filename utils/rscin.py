"""
RsCIN: Re-scoring with Constrained Image-level Neighborhood (MuSc, ICLR 2024).

Post-processing step that refines image-level anomaly scores using CLS token
similarity. Core insight: CLS token distance is robust to local noise, so two
images with high CLS similarity should have similar anomaly scores.

An image is considered normal only if BOTH itself AND its k-nearest neighbors
(by CLS token similarity) have low anomaly scores. This suppresses false
positives from domain-shift noise and amplifies true anomalies.
"""

import torch
import torch.nn.functional as F


def rscin_refine(scores, cls_tokens, k_list=(5, 10, 20)):
    """
    Re-score image-level anomaly scores via CLS token graph propagation.

    Builds a k-NN graph using CLS token similarity, then propagates scores
    through the graph: each image's score is refined by the weighted average
    of its k-nearest neighbors' scores.

    Args:
        scores: [N] initial image-level anomaly scores
        cls_tokens: [N, D] CLS (class) token features from ViT
        k_list: tuple of k values for multi-window mask (default: 5, 10, 20)

    Returns:
        refined: [N] re-scored anomaly scores
    """
    N = scores.shape[0]
    if N < 2:
        return scores.clone()

    # L2-normalize CLS tokens for cosine similarity
    F_norm = F.normalize(cls_tokens.float(), dim=1)       # [N, D]
    W = F_norm @ F_norm.t()                                 # [N, N]

    C = scores.float().unsqueeze(1)                        # [N, 1]
    C_refined = C.clone()

    for k in k_list:
        k_eff = min(k, N - 1)
        if k_eff < 1:
            continue

        # k-NN mask: M_k[i,j] = 1 if j ∈ k-NN(i), else 0
        _, topk_idx = torch.topk(W, k=k_eff, dim=1)       # [N, k_eff]
        M_k = torch.zeros_like(W)
        M_k.scatter_(1, topk_idx, 1.0)

        # D⁻¹ (M_k ⊙ W) C, where D[i,i] = Σ_j (M_k ⊙ W)[i,j]
        W_masked = M_k * W
        D = W_masked.sum(dim=1, keepdim=True).clamp(min=1e-8)
        C_neighbor = (W_masked @ C) / D                    # [N, 1]

        C_refined = C_refined + C_neighbor

    # Eq.5: average over all windows + original
    C_refined = C_refined / (len(k_list) + 1)              # [N, 1]

    return C_refined.squeeze(1)
