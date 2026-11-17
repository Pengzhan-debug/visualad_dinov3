"""
Training utility functions for VisualAD
"""
import torch
import torch.nn as nn


def print_training_parameters(args, logger):
    """Print all training parameters before starting training"""
    total_epochs = getattr(args, "stage1_epochs", 0) + getattr(args, "stage2_epochs", 0)
    logger.info(f"Training: {args.train_dataset} | Backbone: {args.backbone} | "
                f"Stage1 Epochs: {getattr(args, 'stage1_epochs', 'N/A')} | "
                f"Stage2 Epochs: {getattr(args, 'stage2_epochs', 'N/A')} | "
                f"Total Epochs: {total_epochs} | BS: {args.batch_size} | "
                f"Image: {args.image_size} | Layers: {args.features_list}")


def validate_training_setup(args, model, device, logger):
    """Validate training setup and requirements"""
    if isinstance(device, str):
        device = torch.device(device)

    if device.type == 'cuda':
        dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size).to(device)
        with torch.no_grad():
            _ = model.encode_image(dummy_input, args.features_list)
        del dummy_input
        torch.cuda.empty_cache()

    import os
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data path does not exist: {args.train_data_path}")


def setup_model_training(model):
    """Configure model parameters for training"""
    for param in model.parameters():
        param.requires_grad = False

    model.visual.anomaly_token.requires_grad = True
    model.visual.normal_token.requires_grad = True

    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None:
        ln_post.weight.requires_grad = True
        ln_post.bias.requires_grad = True


def create_optimizer(model, layer_transforms, args, cross_attn=None):
    """Create optimizer with different learning rates for different components"""
    optimizer_params = [
        {
            'params': [model.visual.anomaly_token],
            'lr': args.stage1_token_lr,
            'weight_decay': 0.01,
            'name': 'token',
        },
        {
            'params': [model.visual.normal_token],
            'lr': args.stage1_token_lr,
            'weight_decay': 0.01,
            'name': 'token',
        },
    ]

    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None:
        optimizer_params.append({
            'params': [ln_post.weight, ln_post.bias],
            'lr': args.learning_rate,
            'weight_decay': 0.01,
            'name': 'ln_post',
        })

    for transform in layer_transforms.values():
        optimizer_params.append({
            'params': transform.parameters(),
            'lr': args.learning_rate,
            'weight_decay': 0.01,
            'name': 'layer_transform',
        })

    if cross_attn is not None:
        optimizer_params.append({
            'params': cross_attn.parameters(),
            'lr': args.stage1_cross_attn_lr,
            'weight_decay': 0.01,
            'name': 'cross_attn',
        })

    return torch.optim.AdamW(optimizer_params, betas=(0.9, 0.999))


def setup_feature_transforms(features_list, device, feature_dim):
    """Setup SAF (Spatial-Aware Feature) transforms: independent FeatureTransformMLP per layer.

    Matches the gold checkpoint (pure MLP, no LayerNorm, no residual).

    Args:
        features_list: list of layer indices
        device: torch device
        feature_dim: CLIP feature dimension (1024 for ViT-L)
    """
    from .feature_transform import FeatureTransformMLP

    layer_transforms = torch.nn.ModuleDict()

    for layer_idx in features_list:
        mlp = FeatureTransformMLP(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            output_dim=feature_dim,
            dropout=0.0,
        ).to(device)
        layer_transforms[f'layer_{layer_idx}'] = mlp

    return layer_transforms


class _LayerAdapterWrapper(torch.nn.Module):
    """Wrapper that binds a layer index to a shared adapter for ModuleDict compatibility."""
    def __init__(self, adapter, layer_idx):
        super().__init__()
        self.adapter = adapter
        self.layer_idx = layer_idx

    def forward(self, x):
        return self.adapter(x, self.layer_idx)


def check_for_nan(tensor, name, logger, epoch=None):
    """Check tensor for NaN values and log if found"""
    if torch.isnan(tensor).any():
        msg = f"NaN detected in {name}"
        if epoch is not None:
            msg += f" at epoch {epoch+1}"
        logger.error(msg)
        return True
    return False


def compute_segmentation_loss(similarity_map_list, gt, loss_focal, loss_dice):
    """Compute segmentation loss from similarity maps"""
    seg_losses = []
    for similarity_map in similarity_map_list:
        # Hardcoded: beta_focal=1.0, beta_dice=1.0
        seg_losses.append(loss_focal(similarity_map, gt))
        # Only use anomaly channel to avoid double-counting (since channel 0 = 1 - channel 1)
        seg_losses.append(loss_dice(similarity_map[:, 1, :, :], gt))
    return sum(seg_losses) if seg_losses else torch.tensor(0.0, device=gt.device, requires_grad=False)


def validate_gradients(model, logger, epoch):
    """Validate and clip gradients"""
    if model.visual.anomaly_token.grad is not None:
        if check_for_nan(model.visual.anomaly_token.grad, "anomaly_token gradient", logger, epoch):
            return False
        torch.nn.utils.clip_grad_norm_([model.visual.anomaly_token], max_norm=1.0)
    
    if model.visual.normal_token.grad is not None:
        if check_for_nan(model.visual.normal_token.grad, "normal_token gradient", logger, epoch):
            return False
        torch.nn.utils.clip_grad_norm_([model.visual.normal_token], max_norm=1.0)
    
    return True


def save_checkpoint(model, layer_transforms, args, epoch, checkpoint_path,
                    optimizer=None, scheduler=None, spatial_refiner=None,
                    cross_attn=None):
    """Save model checkpoint with full training state for resuming"""
    transform_state_dict = {name: t.state_dict() for name, t in layer_transforms.items()}

    ln_post = getattr(model.visual, "ln_post", None)

    checkpoint_data = {
        "anomaly_token": model.visual.anomaly_token.data.clone(),
        "normal_token": model.visual.normal_token.data.clone(),
        "ln_post_weight": ln_post.weight.data.clone() if ln_post else None,
        "ln_post_bias": ln_post.bias.data.clone() if ln_post else None,
        "features_list": args.features_list,
        "image_size": args.image_size,
        "epoch": epoch,
        "backbone": args.backbone,
        "layer_transforms": transform_state_dict,
        "saf_type": "mlp_simple",
    }

    if spatial_refiner is not None:
        checkpoint_data["spatial_refiner"] = spatial_refiner.state_dict()

    if cross_attn is not None:
        checkpoint_data["cross_attn"] = cross_attn.state_dict()
        checkpoint_data["cross_attn_config"] = {"num_anchors": 4, "dropout": 0.1, "res_scale_init": 0.01}

    if optimizer is not None:
        checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint_data, checkpoint_path)


def load_checkpoint(checkpoint_path, model, layer_transforms,
                    optimizer=None, scheduler=None, device=None,
                    spatial_refiner=None):
    """Load checkpoint and restore full training state for resuming.

    Returns:
        start_epoch (int): epoch to resume from (0-indexed, i.e. already completed epochs)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore tokens
    model.visual.anomaly_token.data = checkpoint["anomaly_token"].to(device)
    model.visual.normal_token.data = checkpoint["normal_token"].to(device)

    # Restore ln_post
    ln_post = getattr(model.visual, "ln_post", None)
    if ln_post is not None and checkpoint.get("ln_post_weight") is not None:
        ln_post.weight.data = checkpoint["ln_post_weight"].to(device)
        ln_post.bias.data = checkpoint["ln_post_bias"].to(device)

    # Restore layer transforms (handles both independent MLPs and shared adapter)
    if "layer_transforms" in checkpoint:
        ltransforms = checkpoint["layer_transforms"]
        if "_shared_adapter" in ltransforms:
            # Shared bottleneck adapter: restore to first wrapper's adapter
            first_transform = next(iter(layer_transforms.values()))
            if hasattr(first_transform, 'adapter'):
                first_transform.adapter.load_state_dict(ltransforms["_shared_adapter"])
        else:
            for name, state_dict in ltransforms.items():
                if name in layer_transforms:
                    layer_transforms[name].load_state_dict(state_dict)

    # Restore spatial refiner
    if spatial_refiner is not None and "spatial_refiner" in checkpoint:
        spatial_refiner.load_state_dict(checkpoint["spatial_refiner"])

    # Restore optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {start_epoch + 1}")
    return start_epoch
