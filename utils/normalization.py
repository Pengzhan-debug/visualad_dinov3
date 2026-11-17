"""
Normalization utilities for anomaly detection data processing
"""
import torch


def normalize_anomaly_maps_per_image(anomaly_maps):
    """
    Normalize each anomaly map independently (per-image normalization)
    Args:
        anomaly_maps: List of anomaly maps [tensor1, tensor2, ...]
    Returns:
        normalized_maps: List of normalized anomaly maps
    """
    normalized_maps = []

    for i, map_tensor in enumerate(anomaly_maps):
        # Get min/max values for current image
        map_min = map_tensor.min().item()
        map_max = map_tensor.max().item()

        # Normalize to [0, 1]
        if map_max > map_min:
            normalized_map = (map_tensor - map_min) / (map_max - map_min)
        else:
            # If all values are the same, set to 0
            normalized_map = torch.zeros_like(map_tensor)

        normalized_maps.append(normalized_map)

    print(f"Completed independent normalization for {len(anomaly_maps)} images")
    return normalized_maps
