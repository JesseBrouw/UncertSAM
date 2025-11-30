import logging
from typing import Dict

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# Note: Shapes in docstrings below follow the convention
# - inputs/logits: [steps, num_objects, H, W]
# - targets:       [num_objects, H, W]
# - ious/logvars:  [steps, num_objects] or [steps, num_objects, H, W]

def bce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy over steps and objects.

    Args:
        inputs: Raw logits of shape [steps, num_objects, H, W].
        targets: Binary mask of shape [num_objects, H, W].

    Returns:
        Scalar BCE loss averaged over spatial dims, summed over steps, normalized by objects.
    """
    assert inputs.ndim == 4 and targets.ndim == 3
    num_objects = inputs.shape[1]
    targets = targets.unsqueeze(0).expand(inputs.shape[0], *targets.shape)   # [steps, num_objects, 1024, 1024]
    
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
    return ce_loss.sum() / num_objects

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2
) -> torch.Tensor: 
    """Sigmoid focal loss (RetinaNet).

    Reference: https://arxiv.org/abs/1708.02002
    Inputs: [steps, num_objects, H, W], targets: [num_objects, H, W].
    """
    assert inputs.ndim == 4 and targets.ndim == 3
    targets = targets.unsqueeze(0).expand(inputs.shape[0], *targets.shape)   # [steps, num_objects, 1024, 1024]
    
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
        
    return loss.mean(dim=(1, 2, 3)).sum()

def iou_loss(
    inputs: torch.Tensor, targets: torch.Tensor, pred_ious: torch.Tensor, use_l1_loss: bool = True
) -> torch.Tensor:
    """Regress predicted IoU to actual IoU computed from binarized logits.

    Args:
        inputs: Raw logits [steps, num_objects, H, W].
        targets: Binary masks [num_objects, H, W].
        pred_ious: Predicted IoUs [steps, num_objects].
        use_l1_loss: If False, use MSE instead.
    """
    assert inputs.ndim == 4 and targets.ndim == 3 
    targets = targets.unsqueeze(0).expand(inputs.shape[0], *targets.shape)   # [steps, num_objects, 1024, 1024]
    
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")

    return loss.mean(dim=1).sum()

def dice_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: 
    """Soft Dice loss over steps and objects.

    Inputs: [steps, num_objects, H, W]; targets: [num_objects, H, W].
    """
    assert inputs.ndim == 4 and targets.ndim == 3
    
    targets = targets.unsqueeze(0).expand(inputs.shape[0], *targets.shape)   # [steps, num_objects, 1024, 1024]
    inputs = inputs.sigmoid()
    
    inputs = inputs.flatten(2)
    targets = targets.flatten(2)
    
    numerator = 2 * (inputs * targets).sum(-1)
    
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1 + 1e-8)
    
    return loss.mean(dim=1).sum()

def sam2_loss(high_res_masks: torch.Tensor, targets: torch.Tensor, ious: torch.Tensor) -> Dict[str, torch.Tensor]: 
    """Combined segmentation loss used for tuning/refinement.

    Returns a dict with individual components and the weighted sum under 'core'.
    """
    dice = dice_loss(high_res_masks.float(), targets.float())
    focal = sigmoid_focal_loss(high_res_masks.float(), targets.float())
    iou = iou_loss(high_res_masks.float(), targets.float(), ious.float())
    
    combo_loss = 20*focal + dice + iou
    
    return {
        'dice' : dice,
        'focal' : focal,
        'iou' : iou,
        'core' : combo_loss
    }

def gaussian_nll(inputs: torch.Tensor, targets: torch.Tensor, log_vars: torch.Tensor) -> torch.Tensor: 
    """Heteroscedastic Gaussian NLL following Kendall & Gal.

    Args:
        inputs: Raw logits [steps, num_objects, H, W].
        targets: Binary masks [num_objects, H, W].
        log_vars: Predicted log-variance maps [steps, num_objects, H, W].
    """
    assert inputs.ndim == 4 and targets.ndim == 3 and log_vars.ndim == 4
    
    log_vars = log_vars.clamp_(min=-5, max=5)
    
    targets = targets.unsqueeze(0).expand(inputs.shape[0], *targets.shape)   # [steps, num_objects, 1024, 1024]
    
    # Get probabilities from logits
    probs = inputs.sigmoid()
    
    # Get main loss
    precision = torch.exp(-log_vars)
    loss = (0.5 * (precision * (probs - targets)**2 + log_vars))
    
    # Average over spatial/object/step dimensions
    return loss.mean(dim=(1, 2, 3)).sum() 
    
def variance_network_loss(high_res_masks: torch.Tensor, targets: torch.Tensor, high_res_logvar_maps: torch.Tensor) -> torch.Tensor:
    """Wrapper for Gaussian NLL used to train the variance head."""
    return gaussian_nll(high_res_masks, targets, high_res_logvar_maps)
