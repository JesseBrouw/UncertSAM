import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torchmetrics.classification import BinaryCalibrationError

EPS = 1e-6

def calc_accuracy_metrics(predicted_masks: torch.Tensor, gt_masks: torch.Tensor, logits: torch.Tensor, dataset_name: str) -> torch.Tensor: 
    # All masks are of the same size, so we calculate everything as one big tensor 
    if predicted_masks.ndim == 4:
        pred, gt, logit = predicted_masks.squeeze(0), gt_masks.squeeze(0), logits.squeeze(0)
    else:
        pred, gt, logit = predicted_masks, gt_masks, logits
    
    if dataset_name == 'BIG':
        pred = F.interpolate(pred.float().unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1).to(pred.dtype)
        gt = F.interpolate(gt.float().unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1).to(gt.dtype)
        logit = F.interpolate(logit.float().unsqueeze(1), scale_factor=0.5, mode='bilinear').squeeze(1).to(logit.dtype)
        
    dilated_roi = get_dilated_roi(gt)
        
    results = []
    with torch.no_grad():
        tn, fp, fn, tp = get_confusion_matrix(pred, gt)
        prec = precision(tp, fp)
        rec = recall(tp, fn)
        
        results = torch.stack([
            IoU(pred, gt),
            dice_score(pred, gt),
            mean_absolute_error(pred, gt),
            balanced_error_rate(tn, fp, fn, tp),
            accuracy(tn, fp, fn, tp),
            prec,
            rec,
            weighted_f_measure(prec, rec),
            IoU(pred*dilated_roi, gt*dilated_roi),
            dice_score(pred*dilated_roi, gt*dilated_roi),
            ], dim=1)
        
        roi_metrics = []
        bound_ious = []
        for m_id in range(pred.shape[0]):
            pred_roi = pred[m_id][dilated_roi[m_id]]
            gt_roi = gt[m_id][dilated_roi[m_id]]
            
            tn, fp, fn, tp = get_confusion_matrix(pred_roi, gt_roi)
            prec = precision(tp, fp)
            rec = recall(tp, fn)
            
            roi_metrics.append(torch.stack([
                mean_absolute_error(pred_roi, gt_roi),
                balanced_error_rate(tn, fp, fn, tp),
                accuracy(tn, fp, fn, tp),
                prec,
                rec,
                weighted_f_measure(prec, rec),
                ]))
        
            pred_mask = pred[m_id]
            gt_mask = gt[m_id]
            bound_ious.append(torch.stack([boundary_iou(gt_mask, pred_mask, dilation_ratio=0.02), boundary_iou(gt_mask, pred_mask, dilation_ratio=0.01)]))
        
        roi_metrics = torch.stack(roi_metrics, dim=0)
        bound_ious = torch.stack(bound_ious, dim=0)
    
    results = torch.cat((results, roi_metrics, bound_ious), dim=1)
    return results.cpu()

         
def calc_uncertainty_metrics(predicted_masks: torch.Tensor, gt_masks: torch.Tensor, logits: torch.Tensor, umaps: Optional[torch.Tensor], std_maps: torch.Tensor, dataset_name: str) -> torch.Tensor:
    if predicted_masks.ndim == 4:
        pred, gt, logit, umaps, std_maps = predicted_masks.squeeze(0), gt_masks.squeeze(0), logits.squeeze(0), umaps.squeeze(0), std_maps.squeeze(0)
    else:
        pred, gt, logit, umaps, std_maps = predicted_masks, gt_masks, logits, umaps, std_maps
    
    if dataset_name == 'BIG':
        pred = F.interpolate(pred.float().unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1).to(pred.dtype)
        gt = F.interpolate(gt.float().unsqueeze(1), scale_factor=0.5, mode='nearest').squeeze(1).to(gt.dtype)
        logit = F.interpolate(logit.float().unsqueeze(1), scale_factor=0.5, mode='bilinear').squeeze(1).to(logit.dtype)
        umaps = F.interpolate(umaps.float().unsqueeze(1), scale_factor=0.5, mode='bilinear').squeeze(1).to(umaps.dtype)
        std_maps = F.interpolate(std_maps.float().unsqueeze(1), scale_factor=0.5, mode='bilinear').squeeze(1).to(std_maps.dtype)
        
    dilated_roi = get_dilated_roi(gt)
    
    bce = BinaryCalibrationError(n_bins=10, norm='l1')
    results = []
    with torch.no_grad():
        for idx in range(pred.shape[0]):
            # Select the portion that corresponds to the original image
            pred_ = pred[idx].unsqueeze(0) # make sure pred & gt are 2D, remove potential batch dims
            gt_ = gt[idx].unsqueeze(0)
            logit_ = logit[idx].unsqueeze(0)
            umap_ = umaps[idx].unsqueeze(0)
            std_map_ = std_maps[idx].unsqueeze(0)
            
            roi_mask = dilated_roi[idx]
            pred_roi = pred[idx][roi_mask]
            gt_roi = gt[idx][roi_mask]
            logit_roi = logit[idx][roi_mask]
            umap_roi = umaps[idx][roi_mask]
            std_roi = std_maps[idx][roi_mask]
            
            probs_ = logit_.sigmoid()
            probs_roi = logit_roi.sigmoid()

            pearson = pearson_corr(umap_.view(-1), torch.abs(gt_ - probs_).view(-1))
            sample_bce = bce(probs_.view(-1), gt_.view(-1))
            sharpness =  torch.mean(umap_)
            brier = brier_score(probs_.reshape(probs_.shape[0], -1), gt_.reshape(gt_.shape[0], -1))
            
            pearson_roi = pearson_corr(umap_roi, torch.abs(gt_roi - probs_roi))
            sample_bce_roi = bce(probs_roi, gt_roi)
            sharpness_roi =  torch.mean(umap_roi)
            brier_roi = brier_score(probs_roi, gt_roi)
            
            inter_sample_std, inter_sample_std_roi = torch.mean(std_map_), torch.mean(std_roi)
            
            u_metrics =  [  
                pearson, 
                sample_bce,
                sharpness,
                brier,
                inter_sample_std,
                pearson_roi,
                sample_bce_roi,
                sharpness_roi,
                brier_roi,
                inter_sample_std_roi
            ]
            
            results.append(torch.tensor(u_metrics))
            
        return torch.stack(results, dim=0).cpu()
    
def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
    x = x.view(-1)
    y = y.view(-1)
    
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    
    centered_x = x - x_mean
    centered_y = y - y_mean
    
    pearson_corr = torch.sum(centered_x*centered_y) / ( (torch.sqrt(torch.sum(centered_x ** 2)) * torch.sqrt(torch.sum(centered_y ** 2)) ) + EPS)
    return pearson_corr


def get_confusion_matrix(preds: torch.Tensor, gts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    preds, gts = preds.bool(), gts.bool()  
    
    if preds.ndim == 1 and gts.ndim == 1:
        # For ROI calculation 
        tp = torch.sum(preds & gts)
        tn = torch.sum(~preds & ~gts)
        fp = torch.sum(preds & ~gts)
        fn = torch.sum(~preds & gts)

        return tn, fp, fn, tp
    else:
        # Use efficient bitwise operations to calculate confusion matrix 
        # number of 1's that overlap (true positives)
        tp = torch.sum(preds & gts, dim=(-2, -1)) 
        # number of 0's that overlap (true negatives)
        tn = torch.sum(~preds & ~gts, dim=(-2, -1))
        # number of predictions that overlap with flipped zero's (false positives)
        fp = torch.sum(preds & ~gts, dim=(-2, -1))
        # number of flipped predictions that overlap with gt (false negatives)
        fn = torch.sum(~preds & gts, dim=(-2, -1))
        
        return tn, fp, fn, tp

def IoU(preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
    preds, gts = preds.bool(), gts.bool()  
    
    intersection = torch.sum(preds & gts, dim=(-2, -1))  # true positives
    union = torch.sum(preds | gts, dim=(-2, -1))
    return intersection / (union + EPS) 

def dice_score(preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor: 
    preds, gts = preds.bool(), gts.bool()  
    
    intersection = torch.sum(preds & gts, dim=(-2, -1))
    denominator = torch.sum(preds, dim=(-2, -1)) + torch.sum(gts, dim=(-2, -1))
    return (2.0 * intersection) / (denominator + EPS)

def balanced_error_rate(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor:
    
    fp_rate = fp / (tn + fp + EPS)
    fn_rate = fn / (tp + fn + EPS)
    
    return 0.5 * (fp_rate + fn_rate)

def accuracy(tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp: torch.Tensor) -> torch.Tensor: 
    return (tp + tn) / (tp + fn + tn + fp + EPS)

def precision(tp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor: 
    return tp / (tp + fp + EPS)
    
def recall(tp: torch.Tensor, fn: torch.Tensor) -> torch.Tensor: 
    return tp / (tp + fn + EPS)

def weighted_f_measure(prec: torch.Tensor, rec: torch.Tensor, beta: int = 1) -> torch.Tensor: 
    beta_sq = beta ** 2
    return ((1 + beta_sq) * prec * rec) / (beta_sq * prec + rec + EPS)

def mean_absolute_error(preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor: 
    if preds.ndim == 1:
        return torch.mean(torch.abs(gts - preds))
    else:
        return torch.mean(torch.abs(gts - preds), dim=(-2, -1))

def entropy(probs: torch.Tensor) -> torch.Tensor: 
    dtype = probs.dtype

    if dtype == torch.bfloat16:
        eps = 1e-2 
    elif dtype == torch.float16:
        eps = 1e-4 
    else: 
        eps = 1e-7
        
    probs = probs.clamp(eps, 1.0 - eps) # make sure the log computation is nan/inf proof
    return -probs * probs.log() - (1.0 - probs) * (1.0 - probs).log()

def safe_logit(probs: torch.Tensor) -> torch.Tensor:
    dtype = probs.dtype
    
    if dtype == torch.bfloat16:
        eps = 1e-2
    elif dtype == torch.float16:
        eps = 1e-4
    else:
        eps = 1e-7
        
    return torch.logit(probs, eps=eps)


def spearman_r_correlation(umaps_flat: torch.Tensor, error_maps_flat: torch.Tensor) -> torch.Tensor:
    device = umaps_flat.device
    umaps_flat = umaps_flat.detach().cpu().numpy().astype(np.float32)
    error_maps_flat = error_maps_flat.detach().cpu().numpy().astype(np.float32)
    
    spearman_corrs = np.array([spearmanr(umaps_flat[i], error_maps_flat[i])[0] for i in range(umaps_flat.shape[0])]) 
    
    return torch.from_numpy(spearman_corrs).to(device)
        
def brier_score(pred_probs_flat: torch.Tensor, gts_flat: torch.Tensor) -> torch.Tensor:
    if pred_probs_flat.ndim == 1 and gts_flat.ndim == 1:
        pred_probs_flat, gts_flat = pred_probs_flat.unsqueeze(0), gts_flat.unsqueeze(0)
        
    brier_scores = torch.mean((pred_probs_flat - gts_flat)**2, dim=1)
    return brier_scores
    
    
def get_dilated_roi(gts: torch.Tensor, dilation_ratio: float = 0.02) -> torch.Tensor: 
    assert gts.ndim == 3
    
    B, H, W = gts.shape 
    diagonal = math.sqrt(H**2 + W**2)
    d = math.ceil(diagonal * dilation_ratio)

    kernel_size = 2 * d + 1   # --> makes sure the convolution keeps the size the same. 
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=gts.device)

    target_region_masks = torch.zeros_like(gts, dtype=torch.bool)

    for m in range(B):
        mask = gts[m].unsqueeze(0).unsqueeze(0).float()  # shape: (1, 1, H, W)
        dilated = F.conv2d(mask, kernel, padding=d).squeeze(0).squeeze(0) > 0
        target_region_masks[m] = dilated

    return target_region_masks


# Function implementation from https://github.com/SysCV/sam-hq/blob/main/train/utils/misc.py 
# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray: 
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt: torch.Tensor, dt: torch.Tensor, dilation_ratio: float = 0.02) -> torch.Tensor: 
    device = gt.device
    dt = (dt>0).cpu().byte().numpy()
    gt = gt.cpu().byte().numpy()

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return torch.tensor(boundary_iou).float().to(device)
