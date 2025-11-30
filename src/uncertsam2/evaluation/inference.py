from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import (
    calc_accuracy_metrics,
    calc_uncertainty_metrics,
    entropy,
    safe_logit,
)
from src.prompt_utils import prepare_prompt_inputs
from src.uncertainsam2 import UncertainSAM2
from src.uncertsam2.visualization import plot_example

log = logging.getLogger(__name__)

def decode_mask(rle: Dict[str, Any]) -> torch.Tensor:
    """Decode a COCO-style RLE mask to a tensor."""
    mask = torch.from_numpy(
        coco_mask.decode({"size": rle["size"], "counts": base64.b64decode(rle["counts"])})
    )
    return mask

def load_targets(
    annotation_mapping: pd.DataFrame,
    image_ids: List[int],
    mask_ids: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Load GT masks for a batch using image and mask identifiers."""
    masks: List[torch.Tensor] = []
    for _, m_id in zip(image_ids, mask_ids):
        ann = annotation_mapping[annotation_mapping.id == m_id]["segmentation"].values[0]
        mask = decode_mask(ann)
        masks.append(mask)

    return torch.stack(masks, dim=0).float().to(device)


def predict_batch(
    model: UncertainSAM2,
    batch: Union[torch.Tensor, List[torch.Tensor]],
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    box_coords: torch.Tensor,
    box_labels: torch.Tensor,
    device: torch.device,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]],
]:
    """Run a forward pass for a batch or a list of TTA samples."""
    if not isinstance(batch, list):
        batch = batch.to(device)
        low_res, umap, std_map, _, prompts = model(
            batch,
            point_coords,
            point_labels,
            box_coords,
            box_labels,
            multistep_inference=False,
        )
    else:
        low_res_list = []
        prompts = None
        for sample in batch:
            sample = sample.to(device)
            point_inputs = [inp[0] for inp in prompts] if prompts is not None else None
            lrm, _, _, _, prompts = model(
                sample,
                point_coords,
                point_labels,
                box_coords,
                box_labels,
                point_inputs=point_inputs,
                multistep_inference=False,
            )
            low_res_list.append(lrm)
        stacked = torch.stack(low_res_list).sigmoid()
        mean = stacked.mean(dim=0)
        std_map = stacked.std(dim=0)
        umap = entropy(mean)
        low_res = safe_logit(mean)

    return low_res, umap, std_map, prompts


def upscale_and_binarise(
    low_res: torch.Tensor,
    umap: Optional[torch.Tensor],
    std_map: Optional[torch.Tensor],
    target_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upscale predictions and binarize logits to masks."""
    up_logits = F.interpolate(low_res, size=target_shape, mode="bilinear").squeeze(1)
    up_umap = F.interpolate(umap.float(), size=target_shape, mode="bilinear").squeeze(1)
    up_std_map = F.interpolate(std_map.float(), size=target_shape, mode="bilinear").squeeze(1)

    up_masks = (up_logits > 0.0).float()
    return up_masks, up_logits, up_umap, up_std_map


def evaluate_dataset(
    dataloader: DataLoader,
    model: UncertainSAM2,
    dataset_name: str,
    data_dir: str,
    annotation_mapping: pd.DataFrame,
    generator: torch.Generator,
    device: torch.device,
    log_dir: str = "./",
    split: str = "test",
) -> pd.DataFrame:
    """Evaluate a dataset, write CSV, and optionally save qualitative examples."""
    start = time.time()

    model = model.to(device).eval()
    log_dir_path = Path(log_dir)

    examples_dir = log_dir_path / "examples" / dataset_name
    examples_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[np.ndarray] = []
    all_identifiers: List[Tuple[int, int]] = []
    plot_every = max(1, len(dataloader) // 20)

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Eval")
        for idx, batch in pbar:
            b = batch[0] if isinstance(batch, list) else batch
            object_ids = b.metadata.unique_objects_identifier.numpy().squeeze(0)
            img_ids = object_ids[:, 0].tolist()
            mask_ids = object_ids[:, 1].tolist()

            target = load_targets(annotation_mapping, img_ids, mask_ids, device)

            target_mask = b.masks
            target_bbox = b.bboxes[0].to(device)

            point_coords, point_labels, box_coords, box_labels = prepare_prompt_inputs(
                bbox_prompts=target_bbox,
                masks=target_mask,
                num_points_to_sample=8,
                device=device,
                generator=generator,
                sampling_strategy="uniform",
                sample_from_gt_probability=1.0,
            )

            low_res, umap, std_map, prompts = predict_batch(
                model,
                batch,
                point_coords,
                point_labels,
                box_coords,
                box_labels,
                device,
            )

            if umap.ndim == 5:
                umap = umap.squeeze(0)

            up_masks, up_logits, up_umap, up_std_map = upscale_and_binarise(
                low_res, umap, std_map, target.shape[-2:]
            )

            acc = calc_accuracy_metrics(up_masks, target, up_logits, dataset_name)
            unc = calc_uncertainty_metrics(
                up_masks, target, up_logits, up_umap, up_std_map, dataset_name
            )

            metrics = torch.cat((acc, unc), dim=1).cpu().numpy()

            acc_mean = acc.mean(dim=0)
            unc_mean = unc.mean(dim=0)
            iou = acc_mean[0].item()
            dice = acc_mean[1].item()
            ece = unc_mean[1].item()
            brier = unc_mean[3].item()
            avg_std = unc_mean[4].item()
            pbar.set_postfix(
                {
                    "IM_ID": img_ids[0],
                    "IoU": f"{iou:.3f}",
                    "dice": f"{dice:.3f}",
                    "ECE": f"{ece:.3f}",
                    "brier": f"{brier:.3f}",
                    "std": f"{avg_std:.3f}",
                }
            )

            all_metrics.append(metrics)
            all_identifiers.extend(zip(img_ids, mask_ids))

            if ((idx + 1) % plot_every == 0):
                img_file = (
                    Path(data_dir) / dataset_name / split / "images" / f"{img_ids[0]:08d}.jpg"
                )
                save_dir = examples_dir / f"{img_ids[0]:08d}"
                os.makedirs(save_dir, exist_ok=True)

                plot_example(
                    img_file,
                    target,
                    up_masks,
                    up_logits,
                    up_umap,
                    up_std_map,
                    prompts,
                    save_dir,
                )

    metrics_arr = np.vstack(all_metrics)
    cols = [
        "IoU",
        "dice",
        "MAE",
        "BER",
        "acc",
        "prec",
        "rec",
        "F1",
        "IoU_",
        "dice_",
        "MAE_",
        "BER_",
        "acc_",
        "prec_",
        "rec_",
        "F1_",
        "BIoU_2",
        "BIoU_1",
        "pearson",
        "ECE",
        "sharp",
        "brier",
        "avg_sample_std",
        "pearson_",
        "ECE_",
        "sharp_",
        "brier_",
        "avg_sample_std_",
    ]

    df = pd.DataFrame(metrics_arr, columns=cols)
    df["image_id"], df["mask_id"] = zip(*all_identifiers)
    df["dataset"] = dataset_name

    df.to_csv(log_dir_path / f"{dataset_name}_{split}.csv", index=False)
    mean_stats = df[cols].mean().round(4)
    log.info(f"Finished evaluating {dataset_name} in {time.time() - start:.1f}s")
    log.info(f"\n{mean_stats.to_string()} \n")

    return df


__all__ = [
    "decode_mask",
    "evaluate_dataset",
    "load_targets",
    "predict_batch",
    "upscale_and_binarise",
]
