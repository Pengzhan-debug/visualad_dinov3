import VisualAD_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataset import Dataset
from utils.logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils.transforms import get_transform
from utils.metrics import compute_metrics
from utils.analysis import analyze_classification_distribution
from utils.visualization import visualize_anomaly_results
from utils.anomaly_detection import generate_anomaly_map_from_tokens
from utils.mutual_scoring import fuse_scores
from utils.mutual_scoring_faithful import mutual_scoring_faithful
from utils.rscin import rscin_refine
from utils.scoring import reduce_anomaly_map, DEFAULT_TOPK_RATIO
from collections import defaultdict


def dino_graph_refine(anomaly_map, dino_features, k=8, alpha=0.3, eps=1e-8):
    """
    SPDL-style DINO patch affinity graph propagation (test-time, no training).

    Uses DINOv3 patch similarity to refine CLIP anomaly maps: similar patches
    in DINO space share anomaly evidence, suppressing texture noise while
    preserving true defect boundaries.

    Args:
        anomaly_map: [H, W] CLIP anomaly map
        dino_features: [1, C, h, w] DINOv3 feature map
        k: top-k neighbors for sparse affinity
        alpha: propagation strength (0=no propagation, 1=full)
    Returns:
        refined: [H, W] refined anomaly map
    """
    H, W = anomaly_map.shape
    d = dino_features.float()
    _, C, h, w = d.shape

    # Resize anomaly map to DINO resolution
    amap_dino = F.interpolate(anomaly_map.unsqueeze(0).unsqueeze(0),
                               size=(h, w), mode='bilinear', align_corners=False)
    amap_flat = amap_dino.view(-1)  # [h*w]

    # Normalize DINO features
    d_flat = d.reshape(C, h * w).transpose(0, 1)  # [h*w, C]
    d_norm = F.normalize(d_flat, dim=-1)           # [h*w, C]

    # Sparse top-k affinity (cosine similarity on DINO features)
    sim = torch.mm(d_norm, d_norm.t())  # [h*w, h*w]
    topk_vals, topk_idx = torch.topk(sim, k=k, dim=-1)
    # Sparse row-normalized affinity matrix
    weights = F.softmax(topk_vals / 0.1, dim=-1)  # [h*w, k]

    # Label propagation: refine = alpha * original + (1-alpha) * neighbor_avg
    neighbor_scores = amap_flat[topk_idx]  # [h*w, k]
    refined_flat = alpha * amap_flat + (1.0 - alpha) * (weights * neighbor_scores).sum(dim=-1)
    refined_dino = refined_flat.view(1, 1, h, w)

    # Resize back to original resolution
    refined = F.interpolate(refined_dino, size=(H, W), mode='bilinear',
                            align_corners=False).squeeze(0).squeeze(0)
    return refined


def consistency_fusion(anomaly_maps_list, eps=1e-8):
    """
    Layer consistency-weighted fusion: layers with spatially coherent anomaly
    maps (clustered high scores = real defects) get higher weight than layers
    with scattered noise.

    Replaces CDVM + DINO weighting. No trainable parameters.

    Args:
        anomaly_maps_list: list of [H, W] anomaly maps from different layers
    Returns:
        fused: [H, W] consistency-weighted fused anomaly map
    """
    n_layers = len(anomaly_maps_list)
    if n_layers == 1:
        return anomaly_maps_list[0]

    M = torch.stack(anomaly_maps_list, dim=0)  # [L, H, W]
    H, W = M.shape[-2:]

    # Per-layer spatial consistency: 1 / (1 + local_variance)
    # High consistency = scores are clustered = likely real defect
    # Low consistency = scores are scattered = likely noise
    local_mean = F.avg_pool2d(M.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
    local_var = F.avg_pool2d((M.unsqueeze(1) - local_mean.unsqueeze(1)) ** 2,
                              kernel_size=5, stride=1, padding=2).squeeze(1)
    consistency = 1.0 / (1.0 + local_var)  # [L, H, W]

    # Mean consistency per layer → softmax weights
    layer_cons = consistency.view(n_layers, -1).mean(dim=-1)  # [L]
    weights = F.softmax(layer_cons / 0.5, dim=0)  # [L], sharpened

    # Weighted sum
    fused = (weights.view(n_layers, 1, 1) * M).sum(dim=0)  # [H, W]
    return fused


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def test(args):
    logger = get_logger(args.save_path)
    device = torch.device(args.device)

    # Load checkpoint (required)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    # Use checkpoint values
    args.backbone = checkpoint.get("backbone", "ViT-L/14@336px")
    args.image_size = checkpoint.get("image_size", 518)
    args.features_list = checkpoint.get("features_list", [6, 12, 18, 24])

    preprocess, target_transform = get_transform(args)

    # Load model
    model, _ = VisualAD_lib.load(args.backbone, device=device)
    model.eval()
    model.to(device)

    feature_dim = model.visual.embed_dim

    # Load trained tokens
    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)

    # Load feature transforms (handles both MLP and shared adapter checkpoints)
    layer_transforms = nn.ModuleDict()
    if "layer_transforms" in checkpoint:
        ltransforms = checkpoint["layer_transforms"]
        if "_shared_adapter" in ltransforms:
            # Shared bottleneck adapter: reconstruct and load
            from utils.feature_transform import SharedBottleneckAdapter
            bn_dim = checkpoint.get("bottleneck_dim", 64)  # backwards compat with old checkpoints
            adapter = SharedBottleneckAdapter(
                embed_dim=feature_dim, bottleneck=bn_dim,
                num_layers=len(args.features_list), init_scale=0.01
            ).to(device)
            adapter.load_state_dict(ltransforms["_shared_adapter"])
            adapter.eval()
            # Wrap for ModuleDict compatibility
            for i, layer_id in enumerate(args.features_list):
                from utils.training_utils import _LayerAdapterWrapper
                wrapper = _LayerAdapterWrapper(adapter, i)
                layer_transforms[f'layer_{layer_id}'] = wrapper
        else:
            # Per-layer MLP checkpoints
            # Check saf_type to choose the right MLP class
            saf_type = checkpoint.get("saf_type", "mlp_simple")  # old ckpts = mlp_simple
            has_norm = any("norm" in k for k in ltransforms[list(ltransforms.keys())[0]].keys())

            if has_norm:
                from utils.feature_transform import ResidualMLPTransform
                MLPClass = ResidualMLPTransform
            else:
                from utils.feature_transform import FeatureTransformMLP
                MLPClass = FeatureTransformMLP

            for layer_name, state_dict in ltransforms.items():
                hidden_dim = state_dict['mlp.0.weight'].shape[0]
                mlp = MLPClass(
                    input_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    output_dim=feature_dim,
                    dropout=0.0,
                ).to(device)
                mlp.load_state_dict(state_dict)
                mlp.eval()
                layer_transforms[layer_name] = mlp

    # Load cross-attention or token offsets
    cross_attn = None
    if "cross_attn" in checkpoint:
        from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
        cfg = checkpoint.get("cross_attn_config", {})
        cross_attn = build_layer_adaptive_cross_attention(
            layers=args.features_list,
            embed_dim=feature_dim,
            num_anchors=cfg.get("num_anchors", 4),
            dropout=cfg.get("dropout", 0.1),
            res_scale_init=cfg.get("res_scale_init", 0.01),
        ).to(device)
        cross_attn.load_state_dict(checkpoint["cross_attn"])
        cross_attn.eval()

    # Load token offsets (used when cross_attn is not available)
    layer_anomaly_offsets = {}
    layer_normal_offsets = {}
    if "layer_anomaly_offsets" in checkpoint:
        for k, v in checkpoint["layer_anomaly_offsets"].items():
            layer_anomaly_offsets[k] = v.to(device)
    if "layer_normal_offsets" in checkpoint:
        for k, v in checkpoint["layer_normal_offsets"].items():
            layer_normal_offsets[k] = v.to(device)

    # Spatial Context Refinement (learnable, identity-init)
    spatial_refiner = None
    if "spatial_refiner" in checkpoint:
        from utils.spatial_refinement import SpatialRefinement
        spatial_refiner = SpatialRefinement().to(device)
        spatial_refiner.load_state_dict(checkpoint["spatial_refiner"])
        spatial_refiner.eval()

    # DINOv3 teacher for fusion guidance (test-time only, no training needed)
    dino_teacher = None
    try:
        from utils.dino_teacher import DINOTeacher
        dino_teacher = DINOTeacher(layers=args.features_list, device=device)
        dino_teacher.eval()
    except Exception:
        pass  # DINOv3 unavailable → fall back to CDVM-only fusion

    # Text-guided anomaly maps disabled — shown to hurt performance (~1% drop)
    text_anomaly_emb = None
    text_normal_emb = None

    # Test dataset
    test_data = Dataset(root=args.test_data_path, transform=preprocess,
                       target_transform=target_transform, dataset_name=args.test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {obj: {'gt_sp': [], 'pr_sp': [], 'imgs_masks': [], 'anomaly_maps': []} for obj in obj_list}

    # Data for analysis
    all_original_images = []
    all_anomaly_maps = []
    all_gt_masks = []
    all_cls_names = []
    all_anomaly_labels = []
    all_img_paths = []

    # Per-class data for mutual scoring (faithful version: DINO patch-level rarity)
    cls_mutual_data = defaultdict(lambda: {
        'topk_scores': [], 'labels': [],
        'dino_features': [], 'fused_maps': [], 'cls_tokens': [],
    })

    for items in tqdm(test_dataloader):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        results[cls_name]['imgs_masks'].append(gt_mask)
        results[cls_name]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            vision_output = model.encode_image(image, args.features_list)
            anomaly_features = vision_output['anomaly_features']
            normal_features = vision_output['normal_features']
            patch_tokens = vision_output['patch_tokens']
            patch_start_idx = vision_output['patch_start_idx']

            # Per-layer token adaptation: cross_attn or token_offsets
            patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]

            if cross_attn is not None:
                adapted_list = cross_attn(
                    anomaly_features, normal_features,
                    patch_features_list, args.features_list
                )
                anomaly_features_list = [adapted['anomaly'] for adapted in adapted_list]
                normal_features_list = [adapted['normal'] for adapted in adapted_list]
            else:
                anomaly_features_list = []
                normal_features_list = []
                for idx, layer_id in enumerate(args.features_list):
                    anom_offset = layer_anomaly_offsets.get(f'layer_{layer_id}', torch.zeros(feature_dim, device=device))
                    norm_offset = layer_normal_offsets.get(f'layer_{layer_id}', torch.zeros(feature_dim, device=device))
                    anomaly_features_list.append(anomaly_features + anom_offset)
                    normal_features_list.append(normal_features + norm_offset)

            # Generate anomaly maps using CLIP features with SAF
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_tokens):
                anomaly_feat_norm = F.normalize(anomaly_features_list[idx], dim=1, eps=1e-8)
                normal_feat_norm = F.normalize(normal_features_list[idx], dim=1, eps=1e-8)

                transform_key = f'layer_{args.features_list[idx]}'
                if transform_key in layer_transforms:
                    B, N, D = patch_feature.shape
                    patch_feature = layer_transforms[transform_key](patch_feature.view(-1, D)).view(B, N, D)

                anomaly_map = generate_anomaly_map_from_tokens(
                    anomaly_feat_norm, normal_feat_norm,
                    patch_feature[:, patch_start_idx:, :],
                    args.image_size
                )
                # Apply spatial refinement if available
                if spatial_refiner is not None:
                    anomaly_map = spatial_refiner(anomaly_map)
                anomaly_map_list.append(anomaly_map)

            # Extract anomaly maps from GPU and move to CPU
            anomaly_map_amap_list = [am[0].cpu() for am in anomaly_map_list]

            # DINO graph refinement (one-shot, no normalization — preserves pixel quality)
            dino_deepest_saved = None
            if dino_teacher is not None:
                try:
                    dino_feat_maps = dino_teacher.get_feature_maps(image)
                    dino_deepest = dino_feat_maps[-1].cpu()  # deepest layer for affinity
                    for i in range(len(anomaly_map_amap_list)):
                        anomaly_map_amap_list[i] = dino_graph_refine(
                            anomaly_map_amap_list[i], dino_deepest, k=8, alpha=0.3)
                    # Save L2-normalized DINO patches for faithful mutual scoring
                    dino_patches = dino_deepest.squeeze(0)  # [C, h, w]
                    dino_patches = dino_patches.reshape(dino_patches.shape[0], -1).T  # [N, C]
                    dino_deepest_saved = F.normalize(dino_patches.float(), dim=-1).half()
                except Exception:
                    pass
            # Consistency-weighted fusion (replaces CDVM + DINO weighting)
            final_anomaly_map = consistency_fusion(anomaly_map_amap_list)
            final_anomaly_map = final_anomaly_map.unsqueeze(0)  # [1, H, W]

            results[cls_name]['anomaly_maps'].append(final_anomaly_map)

            all_original_images.append(image.detach().cpu())
            all_anomaly_maps.append(final_anomaly_map)
            all_gt_masks.append(gt_mask)
            all_cls_names.append(cls_name)
            all_anomaly_labels.append(items['anomaly'].item())
            all_img_paths.append(items['img_path'][0])

            # Store per-class data for mutual scoring (faithful version)
            topk_val = reduce_anomaly_map(final_anomaly_map[0], mode="topk_mean", topk_ratio=0.01)
            cls_mutual_data[cls_name]['topk_scores'].append(topk_val.item())
            cls_mutual_data[cls_name]['labels'].append(items['anomaly'].item())
            if dino_deepest_saved is not None:
                cls_mutual_data[cls_name]['dino_features'].append(dino_deepest_saved)
                cls_mutual_data[cls_name]['fused_maps'].append(final_anomaly_map[0].clone())
            # Store CLS token for RsCIN
            if 'class_features' in vision_output:
                cls_mutual_data[cls_name]['cls_tokens'].append(
                    vision_output['class_features'].detach().cpu().squeeze(0))

    # --- MuSc-style Mutual Scoring (faithful: DINO patch-level rarity) ---
    print(f"\n🔍 Mutual scoring (faithful: DINO patch-level rarity)...")

    try:
        from scipy.stats import spearmanr
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    diag_rows = []
    for cls_name in obj_list:
        data = cls_mutual_data[cls_name]
        n_imgs = len(data['topk_scores'])
        if n_imgs < 2:
            results[cls_name]['pr_sp'] = torch.tensor(data['topk_scores'], dtype=torch.float32)
            continue

        topk_batch = torch.tensor(data['topk_scores'])  # [B]

        # Faithful mutual scoring via DINO patch-level rarity
        dino_feats = data.get('dino_features', [])
        fused_maps = data.get('fused_maps', [])

        if len(dino_feats) == n_imgs:
            # Stack fused maps
            fused_stack = torch.stack(fused_maps)  # [B, H, W]
            # Faithful mutual scoring on GPU
            mutual_batch = mutual_scoring_faithful(
                fused_stack, dino_feats,
                top_m_ratio=0.01, top_k=5,
            )  # [B]
            # Fuse with topk_mean (weight=0: pure mutual only, since topk is anti-correlated on several classes)
            final_scores = fuse_scores(topk_batch, mutual_batch, weight=0.0)  # [B]
        else:
            # No DINO features → fallback to topk_mean
            mutual_batch = torch.zeros(n_imgs)
            final_scores = topk_batch

        # RsCIN: re-score via CLS token graph (MuSc, ICLR 2024)
        cls_tokens_list = data.get('cls_tokens', [])
        if len(cls_tokens_list) == n_imgs and n_imgs >= 3:
            cls_token_batch = torch.stack(cls_tokens_list)  # [B, D]
            final_scores = rscin_refine(final_scores, cls_token_batch, k_list=(5, 10, 20))

        results[cls_name]['pr_sp'] = final_scores.float()

        # Diagnostics
        labels = torch.tensor(data['labels']).float()
        topk_np = topk_batch.numpy()
        mutual_np = mutual_batch.numpy()
        fused_np = final_scores.numpy()
        label_np = labels.numpy()

        if _has_scipy:
            r_topk, p_topk = spearmanr(topk_np, label_np)
            r_mutual, p_mutual = spearmanr(mutual_np, label_np)
            r_fused, p_fused = spearmanr(fused_np, label_np)
        else:
            def _rank_corr(a, b):
                a_rank = np.argsort(np.argsort(a))
                b_rank = np.argsort(np.argsort(b))
                return np.corrcoef(a_rank, b_rank)[0, 1], 0
            r_topk, _ = _rank_corr(topk_np, label_np)
            r_mutual, _ = _rank_corr(mutual_np, label_np)
            r_fused, _ = _rank_corr(fused_np, label_np)

        diag_rows.append([
            cls_name, f"{n_imgs}",
            f"{r_topk:+.3f}", f"{r_mutual:+.3f}", f"{r_fused:+.3f}",
            f"{topk_np.std():.3f}", f"{mutual_np.std():.3f}",
        ])

        # Free GPU memory after each class
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print diagnostic table
    from tabulate import tabulate
    print("\n📊 Score-Label Spearman correlation (per-class):")
    print(tabulate(diag_rows, headers=[
        'Class', 'N', 'r(topk)', 'r(mutual_rarity)', 'r(fused)', 'σ(topk)', 'σ(mutual)'
    ], tablefmt='pipe'))
    if diag_rows:
        avg_r_topk = np.mean([float(r[2]) for r in diag_rows])
        avg_r_mutual = np.mean([float(r[3]) for r in diag_rows])
        avg_r_fused = np.mean([float(r[4]) for r in diag_rows])
        print(f"Mean r: topk={avg_r_topk:+.3f}  mutual_rarity={avg_r_mutual:+.3f}  fused={avg_r_fused:+.3f}")

    # Build flat score list in image order (for analysis compatibility)
    # Track per-class index to maintain order
    cls_idx = defaultdict(int)
    all_scores = []
    for cls_name in all_cls_names:
        scores = results[cls_name]['pr_sp']
        idx = cls_idx[cls_name]
        all_scores.append(scores[idx].item())
        cls_idx[cls_name] = idx + 1
    fused_scores = all_scores  # for analysis compatibility

    # Normalize for visualization
    from utils.normalization import normalize_anomaly_maps_per_image
    normalized_anomaly_maps = normalize_anomaly_maps_per_image(all_anomaly_maps)
    compute_metrics(results, obj_list, logger)

    # Analysis (optional)
    if args.enable_analysis:
        analysis_dir = os.path.join(args.save_path, 'analysis')
        analyze_classification_distribution(fused_scores, all_cls_names, all_anomaly_labels, analysis_dir)
        visualize_anomaly_results(
            all_original_images, normalized_anomaly_maps, all_gt_masks, fused_scores,
            all_cls_names, all_img_paths, all_anomaly_labels, args.test_dataset, analysis_dir
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VisualAD Test", add_help=True)
    parser.add_argument("--test_data_path", type=str,default='/media/lenovo/sda7/pz/data/br35h', help="test dataset path")
    parser.add_argument("--save_path", type=str, default='/media/lenovo/sda6/data/pz/visualad_dinov3/experiment/train_visa_stage3_6/br35h', help='path to save test results')
    parser.add_argument("--test_dataset", type=str, default='br35h', help="test dataset name")
    parser.add_argument("--checkpoint_path", type=str, default='/media/lenovo/sda6/data/pz/visualad_dinov3/experiment/train_visa_stage3_6/final_model.pth', help="path to trained model checkpoint")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian filter sigma")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
    parser.add_argument("--enable_analysis", action="store_true", help="enable data analysis and visualization")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
