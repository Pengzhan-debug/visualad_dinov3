import VisualAD_lib
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # DINOv3 download mirror
import torch
from torch.cuda.amp import autocast
import argparse
import torch.nn.functional as F
from utils.loss import FocalLoss, BinaryDiceLoss
from utils.dino_teacher import DINOTeacher
from dataset import Dataset
from utils.logger import get_logger
from utils.training_utils import (
    print_training_parameters, validate_training_setup, setup_model_training,
    create_optimizer, setup_feature_transforms, check_for_nan,
    compute_segmentation_loss, validate_gradients, save_checkpoint, load_checkpoint
)
from tqdm import tqdm
import numpy as np
import os
import random
from utils.transforms import get_transform
from utils.scoring import reduce_anomaly_map, DEFAULT_TOPK_RATIO

torch.use_deterministic_algorithms(True, warn_only=False)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Additional deterministic settings
    torch.use_deterministic_algorithms(True, warn_only=True)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def _weighted_focal_loss(logit, target, weight, gamma=2.0):
    """Per-pixel Focal Loss with spatial weights. Weight shape [B,1,H,W]."""
    B, C, H, W = logit.shape
    logit = logit.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
    target = target.reshape(-1).long()
    weight = weight.reshape(-1)                        # [B*H*W]

    log_probs = torch.nn.functional.log_softmax(logit, dim=-1)
    probs = torch.nn.functional.softmax(logit, dim=-1)
    pt = probs[torch.arange(logit.shape[0], device=logit.device), target]
    ce = -log_probs[torch.arange(logit.shape[0], device=logit.device), target]
    focal_weight = (1.0 - pt).pow(gamma)
    loss = (focal_weight * ce * weight).sum() / (weight.sum() + 1e-8)
    return loss


def generate_anomaly_map_from_tokens(anomaly_features, normal_features, patch_tokens, image_size):
    """
    Generate pixel-level anomaly map using token features with numerical stability.
    Args:
        anomaly_features: [B, dim] - anomaly token features
        normal_features: [B, dim] - normal token features
        patch_tokens: [B, num_patches, dim] - patch token features
        image_size: target image size
    Returns:
        anomaly_map: [B, H, W] - pixel-level anomaly map
    """
    B = anomaly_features.shape[0]

    # Normalize all features to prevent numerical instability in cosine similarity
    anomaly_features_norm = F.normalize(anomaly_features, dim=1, eps=1e-8)
    normal_features_norm = F.normalize(normal_features, dim=1, eps=1e-8)
    patch_tokens_norm = F.normalize(patch_tokens, dim=2, eps=1e-8)

    # Compute similarity between each patch and anomaly/normal tokens
    anomaly_sim = torch.cosine_similarity(
        patch_tokens_norm, anomaly_features_norm.unsqueeze(1), dim=2
    )  # [B, num_patches]

    normal_sim = torch.cosine_similarity(
        patch_tokens_norm, normal_features_norm.unsqueeze(1), dim=2
    )  # [B, num_patches]

    # Anomaly score = anomaly_similarity - normal_similarity
    anomaly_score = anomaly_sim - normal_sim  # [B, num_patches]

    # Check for NaN in anomaly scores
    if torch.isnan(anomaly_score).any():
        print(f"Warning: NaN detected in anomaly_score")
        anomaly_score = torch.nan_to_num(anomaly_score, nan=0.0)

    # Reshape to spatial dimensions
    num_patches = anomaly_score.shape[1]
    patch_size_from_model = int(np.sqrt(num_patches))

    anomaly_map = anomaly_score.reshape(B, patch_size_from_model, patch_size_from_model)

    # Resize to target image size
    anomaly_map = F.interpolate(
        anomaly_map.unsqueeze(1),
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)

    return anomaly_map


def compute_classification_loss_V2(anomaly_maps_list, labels, device):
    """
    Compute classification loss aligned with test.py inference
    Uses segmentation scores directly without normalization (training doesn't need class-wise normalization)
    Args:
        anomaly_maps_list: List of anomaly maps from different layers [[B, H, W], [B, H, W], ...]
        labels: Ground truth labels [B]
        device: Device
    Returns:
        loss: Binary cross-entropy loss
    """
    if not anomaly_maps_list:
        return torch.tensor(0.0, device=device)

    # Sum anomaly maps from all layers (same as test.py)
    final_anomaly_maps = torch.stack(anomaly_maps_list).sum(dim=0)  # [B, H, W]

    # Reduce anomaly map with Top-K mean to stabilize classification score
    seg_scores = reduce_anomaly_map(
        final_anomaly_maps,
        mode="topk_mean",
        topk_ratio=DEFAULT_TOPK_RATIO
    )  # [B]
    labels_float = labels.float().to(device)
    loss = F.binary_cross_entropy_with_logits(seg_scores, labels_float)

    return loss


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True


def set_stage_learning_rates(optimizer, stage_name, args):
    for param_group in optimizer.param_groups:
        group_name = param_group.get("name", "")
        if group_name == "token":
            param_group["lr"] = args.stage1_token_lr if stage_name == "Stage 1 (token separation)" else 0.0


def set_stage_trainability(model, stage_name, cross_attn):
    for param in model.parameters():
        param.requires_grad = False

    if stage_name == "Stage 1 (token separation)":
        model.visual.anomaly_token.requires_grad = True
        model.visual.normal_token.requires_grad = True
        ln_post = getattr(model.visual, "ln_post", None)
        if ln_post is not None:
            ln_post.weight.requires_grad = True
            ln_post.bias.requires_grad = True
    else:
        model.visual.anomaly_token.requires_grad = False
        model.visual.normal_token.requires_grad = False
        ln_post = getattr(model.visual, "ln_post", None)
        if ln_post is not None:
            ln_post.weight.requires_grad = True
            ln_post.bias.requires_grad = True

    # cross_attn trainable in Stage 2 only (Stage 1 = token separation only)
    if cross_attn is not None:
        for p in cross_attn.parameters():
            p.requires_grad = (stage_name != "Stage 1 (token separation)")


def train(args):
    logger = get_logger(args.save_path)
    device = args.device

    # Load and setup model
    try:
        model, _ = VisualAD_lib.load(args.backbone, device=device)
    except RuntimeError as exc:
        logger.error(f"Failed to load backbone {args.backbone}: {exc}")
        raise

    model.train()
    model.to(device)

    preprocess, target_transform = get_transform(args)

    print_training_parameters(args, logger)

    validate_training_setup(args, model, device, logger)

    # Token adapter: spatial cross-attention (fixed)
    feature_dim = model.visual.embed_dim
    from utils.spatial_cross_attention import build_layer_adaptive_cross_attention
    cross_attn = build_layer_adaptive_cross_attention(
        layers=args.features_list,
        embed_dim=feature_dim,
        num_anchors=4,
        dropout=0.1,
        res_scale_init=0.01,
    ).to(device)
    cross_attn.train()
    logger.info(f"Cross-attention loaded ({sum(p.numel() for p in cross_attn.parameters()):,} params)")

    # Spatial Context Refinement: learnable 3×3 conv to use neighborhood context
    # Initialized as identity — refines anomaly maps by boosting isolated defects
    # and suppressing scattered texture noise (10 params total)
    from utils.spatial_refinement import SpatialRefinement
    spatial_refiner = SpatialRefinement(init_alpha=0.1).to(device)
    spatial_refiner.train()
    logger.info("SpatialRefinement module loaded (10 params, identity-init)")

    # DINOv3 Structural Teacher: provides regularity maps for seg_loss weighting
    dino_teacher = DINOTeacher(layers=args.features_list, device=device)
    dino_teacher.eval()
    logger.info("DINOv3 teacher loaded (frozen, regularity-weighted seg_loss)")

    # Tokens initialized randomly (nn.Parameter defaults in VisionTransformer)
    # Text-based initialization removed — experiments showed it hurts (87.5% vs 89.1% baseline)
    # with random init, as the narrow defect descriptions don't generalize to MVTec's
    # diverse anomaly types.
    logger.info("Tokens initialized randomly (text init removed — proven harmful on MVTec)")

    # Load dataset
    train_data = Dataset(root=args.train_data_path, transform=preprocess,
                         target_transform=target_transform, dataset_name=args.train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True)

    # Setup feature transforms and model training
    layer_transforms = setup_feature_transforms(args.features_list, device, feature_dim)

    setup_model_training(model)

    optimizer = create_optimizer(model, layer_transforms, args, cross_attn=cross_attn)
    # Add spatial refiner to optimizer
    optimizer.add_param_group({
        'params': spatial_refiner.parameters(),
        'lr': args.learning_rate,
        'weight_decay': 0.0,  # No WD for 10-param module
        'name': 'spatial_refiner',
    })
    token_param_ids = {id(model.visual.anomaly_token), id(model.visual.normal_token)}
    for param_group in optimizer.param_groups:
        group_param_ids = {id(p) for p in param_group["params"]}
        if group_param_ids & token_param_ids:
            param_group["name"] = "token"

    stage1_epochs = args.stage1_epochs
    stage2_epochs = args.stage2_epochs
    total_epochs = stage1_epochs + stage2_epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs,
                                                           eta_min=args.learning_rate * 0.1)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, layer_transforms,
                                      optimizer=optimizer, scheduler=scheduler, device=device)
        logger.info(f"Resumed from {args.resume}, starting from epoch {start_epoch + 1}/{total_epochs}")
    else:
        # Only setup initial stage if not resuming
        initial_stage_name = "Stage 2 (SCA/SAF training)" if stage1_epochs <= 0 else "Stage 1 (token separation)"
        set_stage_learning_rates(optimizer, initial_stage_name, args)

    amp_enabled = True
    amp_dtype = torch.bfloat16

    # Initialize losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    for epoch in tqdm(range(start_epoch, total_epochs)):
        # Stage scheduling: Stage 1 freezes cross-attention, Stage 2 unfreezes it
        if epoch < stage1_epochs:
            stage_name = "Stage 1 (token separation)"
        else:
            stage_name = "Stage 2 (SCA/SAF training)"

        set_stage_trainability(model, stage_name, cross_attn)
        set_stage_learning_rates(optimizer, stage_name, args)
        model.train()

        logger.info(f"Epoch {epoch + 1}/{total_epochs} - {stage_name}")

        loss_list = []
        image_loss_list = []
        sep_loss_list = []

        for items in tqdm(train_dataloader):
            image = items['img'].to(device)
            label = items['anomaly']
            # Squeeze only the channel dimension (dim=1), preserve batch dimension
            gt = items['img_mask'].squeeze(1).to(device)  # [B, 1, H, W] -> [B, H, W]
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            def compute_losses():
                vision_output = model.encode_image(image, args.features_list)
                anomaly_features = vision_output['anomaly_features']
                normal_features = vision_output['normal_features']
                patch_tokens = vision_output['patch_tokens']
                patch_start_idx = vision_output['patch_start_idx']

                # Per-layer token adaptation: spatial cross-attention
                patch_features_list = [pt[:, patch_start_idx:, :] for pt in patch_tokens]

                adapted_list = cross_attn(
                    anomaly_features, normal_features,
                    patch_features_list, args.features_list
                )
                anomaly_features_list = [adapted['anomaly'] for adapted in adapted_list]
                normal_features_list = [adapted['normal'] for adapted in adapted_list]

                # Separation loss on the last-layer token features
                final_anomaly_features = anomaly_features_list[-1]
                final_normal_features = normal_features_list[-1]

                final_anomaly_features_norm = F.normalize(final_anomaly_features, dim=1, eps=1e-8)
                final_normal_features_norm = F.normalize(final_normal_features, dim=1, eps=1e-8)

                if (check_for_nan(final_anomaly_features_norm, "normalized anomaly features", logger, epoch) or
                        check_for_nan(final_normal_features_norm, "normalized normal features", logger, epoch)):
                    return None

                # Squared hinge: ReLU(cos+0.5)² → pushes cos below -0.5 (≥120°).
                sep_val = F.relu(F.cosine_similarity(
                    final_anomaly_features_norm, final_normal_features_norm, dim=-1) + 0.5).pow(2).mean()
                if check_for_nan(sep_val, "token_separation_loss", logger, epoch):
                    return None

                # Merged loop: Generate both Segmentation Maps and Classification Maps
                # Avoid duplicate computation, improve training speed
                similarity_map_list = []
                anomaly_maps_list = []
                clip_feat_maps = []  # [B, C, h, w] for L_struct

                for idx_layer, patch_feature in enumerate(patch_tokens):
                    anomaly_feat_norm = F.normalize(anomaly_features_list[idx_layer], dim=1, eps=1e-8)
                    normal_feat_norm = F.normalize(normal_features_list[idx_layer], dim=1, eps=1e-8)

                    current_layer = args.features_list[idx_layer]
                    transform_key = f'layer_{current_layer}'

                    # SAF transform on CLIP patch features
                    if transform_key in layer_transforms:
                        batch_size, num_patches, feat_dim = patch_feature.shape
                        patch_feature_flat = patch_feature.view(-1, feat_dim)
                        transformed_feature = layer_transforms[transform_key](patch_feature_flat)
                        patch_feature = transformed_feature.view(batch_size, num_patches, feat_dim)

                    # Collect feature map for L_struct (after SAF, CLIP patches only)
                    patch_feat = patch_feature[:, patch_start_idx:, :]  # [B, N_p, D]
                    h = w = int(patch_feat.shape[1] ** 0.5)
                    clip_feat_maps.append(patch_feat.transpose(1, 2).reshape(batch_size, feat_dim, h, w))

                    # Generate anomaly map
                    anomaly_map = generate_anomaly_map_from_tokens(
                        anomaly_feat_norm, normal_feat_norm,
                        patch_feat, args.image_size
                    )

                    # Spatial context refinement: boost isolated defects,
                    # suppress scattered texture noise (learnable, identity-init)
                    anomaly_map = spatial_refiner(anomaly_map)

                    # For Segmentation Loss
                    anomaly_map_sigmoid = torch.sigmoid(anomaly_map)
                    similarity_map = torch.stack([1 - anomaly_map_sigmoid, anomaly_map_sigmoid], dim=1)
                    similarity_map_list.append(similarity_map)

                    # For Classification Loss (reuse same anomaly_map)
                    anomaly_maps_list.append(anomaly_map)

                # Compute seg_loss with DINOv3 regularity weighting
                # Texture regions (high DINO regularity) get higher loss weight,
                # forcing CLIP to focus on learning texture anomaly discrimination.
                # DINOv3 only provides loss weights — no feature-space alignment.
                seg_val = torch.tensor(0.0, device=device)
                if similarity_map_list and (anomaly_features.requires_grad or normal_features.requires_grad):
                    if args.dino_reg_weight > 0:
                        with torch.no_grad():
                            dino_feat_maps = dino_teacher.get_feature_maps(image)
                            d_last = dino_feat_maps[-1].float()
                            local_mean = F.avg_pool2d(d_last, kernel_size=7, stride=1, padding=3)
                            local_var = F.avg_pool2d((d_last - local_mean) ** 2, kernel_size=7, stride=1, padding=3)
                            var = local_var.mean(dim=1, keepdim=True)
                            reg = 1.0 / (1.0 + var)  # [B, 1, h, w], high = structured texture
                            reg = F.interpolate(reg, size=gt.shape[-2:], mode='bilinear', align_corners=False)
                            reg = reg / (reg.mean() + 1e-8)  # normalize so avg weight = 1
                        for similarity_map in similarity_map_list:
                            w = reg.detach()
                            seg_val += _weighted_focal_loss(similarity_map, gt, w, gamma=2.0)
                            seg_val += loss_dice(similarity_map[:, 1, :, :] * w.squeeze(1), gt * w.squeeze(1))
                    else:
                        seg_val = compute_segmentation_loss(similarity_map_list, gt, loss_focal, loss_dice)

                # Image-level classification via topk BCE (same as Stage 2 / test.py).
                # Uses anomaly map directly — no class token dependency.
                cls_val = compute_classification_loss_V2(anomaly_maps_list, label, device)

                if stage_name == "Stage 1 (token separation)":
                    # Stage 1: sep_loss + seg_loss
                    # sep_loss pushes tokens apart (squared hinge, always has gradient)
                    # seg_loss gives tokens grounding in what anomalies look like
                    if sep_val.requires_grad:
                        loss_components = [args.stage1_sep_weight * sep_val]
                        if seg_val is not None and seg_val.requires_grad:
                            loss_components.append(0.1 * seg_val)
                        total = sum(loss_components)
                        return total, (seg_val if seg_val.requires_grad else torch.tensor(0.0, device=device)), torch.tensor(0.0, device=device), sep_val
                    logger.error("Stage 1 separation loss does not require gradients!")
                    return None

                image_val = compute_classification_loss_V2(anomaly_maps_list, label, device)
                if check_for_nan(image_val, "image_loss", logger, epoch):
                    return None

                loss_components = []
                if image_val.requires_grad:
                    loss_components.append(image_val)
                if seg_val.requires_grad:
                    loss_components.append(seg_val)
                # sep_loss excluded from Stage 2: token directions are already
                # frozen from Stage 1 and cross_attn's residual (scale=0.01)
                # cannot meaningfully corrupt them.

                if not loss_components:
                    logger.error("No loss component requires gradients!")
                    return None

                total = sum(loss_components)
                return total, seg_val, image_val, sep_val

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                result = compute_losses()

            if result is None:
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss, seg_loss_val, image_loss_val, sep_loss_val = result

            seg_loss_value = float(seg_loss_val.detach().item())
            image_loss_value = float(image_loss_val.detach().item())
            sep_loss_value = float(sep_loss_val.detach().item())

            total_loss.backward()

            # Validate gradients after backward pass
            if not validate_gradients(model, logger, epoch):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()

            if check_for_nan(model.visual.anomaly_token, "anomaly_token after update", logger, epoch) or \
                    check_for_nan(model.visual.normal_token, "normal_token after update", logger, epoch):
                break

            loss_list.append(seg_loss_value)
            image_loss_list.append(image_loss_value)
            sep_loss_list.append(sep_loss_value)

        scheduler.step()

        # Log training progress
        if (epoch + 1) % args.print_freq == 0:
            logger.info(
                f'Epoch [{epoch + 1}/{total_epochs}] - {stage_name} - '
                f'seg: {np.mean(loss_list):.4f}, '
                f'cls: {np.mean(image_loss_list):.4f}, '
                f'sep: {np.mean(sep_loss_list):.4f}'
            )

        # Save model checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, layer_transforms, args, epoch + 1,
                            os.path.join(args.save_path, f'epoch_{epoch + 1}.pth'),
                            optimizer=optimizer, scheduler=scheduler,
                            spatial_refiner=spatial_refiner,
                            cross_attn=cross_attn)

    # Save final model
    final_ckp_path = os.path.join(args.save_path, 'final_model.pth')
    save_checkpoint(model, layer_transforms, args, total_epochs, final_ckp_path,
                    optimizer=optimizer, scheduler=scheduler,
                    spatial_refiner=spatial_refiner,
                    cross_attn=cross_attn)

    logger.info(f'Training completed! Model saved to {final_ckp_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VisualAD Training V2", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="/media/lenovo/sda7/pz/data/mvtec",
                        help="train dataset path (must contain meta.json)")
    parser.add_argument("--save_path", type=str, default='/media/lenovo/sda6/data/pz/visualad_dinov3/experiment/train_mvtec_stage3_6', help='path to save results')
    parser.add_argument("--train_dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--backbone", type=str, default="ViT-L/14@336px",
                        choices=VisualAD_lib.available_models(), help="CLIP backbone to use")
    parser.add_argument("--features_list", type=int, nargs="*", default=[6, 12, 18, 24],
                        help="Feature layers to extract")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="epochs for stage 1 token separation")
    parser.add_argument("--stage2_epochs", type=int, default=6, help="epochs for stage 2 SCA/SAF training")
    parser.add_argument("--stage1_sep_weight", type=float, default=0.1, help="token separation loss weight in stage 1")
    parser.add_argument("--stage1_token_lr", type=float, default=0.0005, help="token lr in stage 1")
    parser.add_argument("--stage1_cross_attn_lr", type=float, default=0.0, help="cross-attn lr in stage 1")
    parser.add_argument("--stage2_cross_attn_lr", type=float, default=0.0001, help="cross-attn lr in stage 2")
    parser.add_argument("--dino_reg_weight", type=float, default=0.0, help="DINOv3 regularity-weighted seg_loss (0=disabled). Texture regions get higher loss weight.")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume training from")

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)
    train(args)
