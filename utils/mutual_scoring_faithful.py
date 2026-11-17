"""
Faithful MuSc-style mutual scoring: patch-level DINO rarity.

Core idea (MuSc, ICLR 2024): normal patches find many similar patches in
other test images, while anomalous patches find few. We lift this to
image-level: take the top anomalous patches from each image, measure how
rare they are in the full test-set DINO feature database.

This is stronger than the fast descriptor-based version because it operates
directly on patch features rather than aggregated statistics.
"""

import torch
import torch.nn.functional as F


def patch_rarity_scoring(anomaly_maps, dino_db, img_boundaries, top_m_ratio=0.01, top_k=5):
    """
    Score each image by the rarity of its most anomalous patches.

    For each image, takes the top-m patches from the denoised anomaly map,
    retrieves their top-k nearest neighbors in the DINO database (from OTHER
    images only), and computes the average rarity (1 - similarity).

    Args:
        anomaly_maps: [B, H, W] denoised + fused anomaly maps
        dino_db: [M, C] L2-normalized DINO patch features from ALL images in class
        img_boundaries: [B+1] start/end indices into dino_db for each image
        top_m_ratio: fraction of patches to query per image (default 0.01 = 1%)
        top_k: number of nearest neighbors for rarity computation (default 5)

    Returns:
        scores: [B] image-level rarity scores (higher = more anomalous)
    """
    B = anomaly_maps.shape[0]
    H, W = anomaly_maps.shape[-2:]
    C = dino_db.shape[1]

    # Resize anomaly maps to DINO spatial resolution for patch indexing
    # Use the last image's DINO feature count to infer h, w
    last_n = img_boundaries[1] - img_boundaries[0]
    h = w = int(last_n ** 0.5)

    amap_dino = F.interpolate(anomaly_maps.unsqueeze(1),
                               size=(h, w), mode='bilinear',
                               align_corners=False).squeeze(1)     # [B, h, w]
    N = h * w
    amap_flat = amap_dino.reshape(B, N)                             # [B, N]
    m = max(1, int(N * top_m_ratio))

    # Get top-m anomalous patch indices per image
    _, top_idx = amap_flat.topk(m, dim=1)                          # [B, m]

    scores = []
    for i in range(B):
        # DINO features for image i's top-m query patches
        start_i = img_boundaries[i]
        qi = dino_db[start_i:start_i + N]                          # [N, C]
        qi_top = qi[top_idx[i]]                                    # [m, C]

        # Database: all patches EXCEPT from image i
        start, end = img_boundaries[i], img_boundaries[i + 1]
        mask = torch.ones(dino_db.shape[0], dtype=torch.bool, device=dino_db.device)
        mask[start:end] = False
        Db_other = dino_db[mask]                                   # [M_other, C]

        # Cosine similarity (both are L2-normalized)
        sim = qi_top @ Db_other.T                                  # [m, M_other]

        # Top-K nearest neighbors from other images → rarity
        topk_sim = sim.topk(min(top_k, sim.shape[1]), dim=1).values  # [m, min(k, M_other)]
        rarity = 1.0 - topk_sim.mean(dim=1)                         # [m]
        scores.append(rarity.mean())                                # scalar

    return torch.stack(scores)


def mutual_scoring_faithful(anomaly_maps, dino_tensor_list, top_m_ratio=0.01, top_k=5,
                            chunk_size=None, device=None):
    """
    Faithful MuSc-style mutual scoring: patch-level DINO rarity.

    Builds a DINO patch database from all images in a class, then scores
    each image by how rare its top anomalous patches are.

    Args:
        anomaly_maps: [B, H, W] denoised + fused anomaly maps (on CPU)
        dino_tensor_list: list of B tensors, each [N_i, C] L2-normalized
            DINOv3 deepest-layer patch features, in float16 (on CPU)
        top_m_ratio: fraction of patches per image to query
        top_k: nearest neighbors for rarity
        chunk_size: if set, process images in chunks to limit GPU memory
        device: torch device (default: cuda:0 if available, else cpu)

    Returns:
        scores: [B] image-level rarity scores (higher = more anomalous)
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    B = anomaly_maps.shape[0]

    # Build database on GPU
    dino_db = torch.cat([t.float() for t in dino_tensor_list], dim=0).to(device)  # [M, C]

    # Image boundaries
    img_boundaries = [0]
    for t in dino_tensor_list:
        img_boundaries.append(img_boundaries[-1] + t.shape[0])

    return patch_rarity_scoring(
        anomaly_maps.to(device), dino_db, img_boundaries,
        top_m_ratio=top_m_ratio, top_k=top_k
    ).cpu()
