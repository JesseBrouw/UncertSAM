from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from PIL import Image


def show_mask(
    mask: np.ndarray,
    ax: Axes,
    color: Optional[List[float]] = None,
    opacity: float = 0.75,
) -> List[float]:
    """Overlay a segmentation mask on an axis and return the used color."""
    color = color if color else (np.random.rand(3).tolist() + [opacity])
    h, w = mask.shape
    mask_img = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
    ax.imshow(mask_img)
    return color


def show_box(box: np.ndarray, ax: Axes) -> None:
    """Draw a bounding box on an axis."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    rect = plt.Rectangle(
        (x0, y0), width, height, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2.5
    )
    ax.add_patch(rect)


def show_points(coords: np.ndarray, labels: np.ndarray, ax: Axes, marker_size: int = 375) -> None:
    """Scatter positive/negative point prompts on an axis."""
    pos = coords[labels == 1]
    neg = coords[labels == 0]

    if len(pos):
        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
    if len(neg):
        ax.scatter(
            neg[:, 0],
            neg[:, 1],
            c="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )


def show_masks(
    image: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    point_coords: Optional[np.ndarray] = None,
    box_coords: Optional[np.ndarray] = None,
    input_labels: Optional[np.ndarray] = None,
    borders: bool = True,
) -> None:
    """Visualize SAM predictions alongside prompts."""
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask.squeeze(), plt.gca())
        if point_coords is not None and input_labels is not None:
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()


def visualise_example(
    images: List[np.ndarray],
    boxes: Optional[List[np.ndarray]],
    point_prompts: Optional[List[np.ndarray]],
    point_labels: Optional[List[np.ndarray]],
    masks: List[np.ndarray],
) -> None:
    """Render multiple prompts/masks for quick qualitative inspection."""
    boxes = [[] for _ in images] if boxes is None else boxes
    point_prompts = [[] for _ in images] if point_prompts is None else point_prompts
    point_labels = [[] for _ in images] if point_labels is None else point_labels

    for im, bxs, points, labels, ms in zip(images, boxes, point_prompts, point_labels, masks):
        plt.figure(figsize=(10, 10))
        plt.imshow(im)
        for mask in ms:
            show_mask(mask.squeeze(), plt.gca(), opacity=0.6)
        for box in bxs:
            # If one image contains multiple bounding boxes, plot separately
            if len(box.shape) > 1:
                for b in box:
                    show_box(b.squeeze(), plt.gca())
            else:
                show_box(box.squeeze(), plt.gca())
        for p, l in zip(points, labels):
            show_points(coords=p, labels=l, ax=plt.gca(), marker_size=150)
        plt.show()


def plot_example(
    img_path: Path,
    target: torch.Tensor,
    pred: torch.Tensor,
    logit: torch.Tensor,
    umap: torch.Tensor,
    std_map: torch.Tensor,
    prompts: List[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]],
    save_path: Path,
) -> None:
    """Create qualitative plots for predictions, errors, prompts and uncertainty."""

    def _fix_ndim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.squeeze(0) if tensor.ndim == 4 else tensor

    target = _fix_ndim(target)
    pred = _fix_ndim(pred)
    logit = _fix_ndim(logit)
    std_map = _fix_ndim(std_map)
    umap = _fix_ndim(umap)

    img = Image.open(img_path)
    orig_w, orig_h = img.size
    black_background = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    im_arr = np.array(img)
    summed_residuals = np.zeros(im_arr.shape[:2])

    fig_pred, pred_ax = plt.subplots(figsize=(12, 12))
    pred_ax.axis("off")
    pred_ax.imshow(black_background)

    fig_gt, gt_ax = plt.subplots(figsize=(12, 12))
    gt_ax.axis("off")
    gt_ax.imshow(black_background)

    fig_err, err_ax = plt.subplots(figsize=(12, 12))
    err_ax.axis("off")
    err_ax.imshow(black_background)

    fig_prompt, prompt_ax = plt.subplots(figsize=(12, 12))
    prompt_ax.axis("off")
    prompt_ax.imshow(img)

    fig_unc, unc_ax = plt.subplots(figsize=(12, 12))
    unc_ax.axis("off")
    fig_res, res_ax = plt.subplots(figsize=(12, 12))
    res_ax.axis("off")

    fig_std, std_ax = plt.subplots(figsize=(12, 12))
    std_ax.axis("off")

    scale_x = (orig_w - 1) / (1024 - 1)
    scale_y = (orig_h - 1) / (1024 - 1)

    colors = None
    for (sparse_prompt, dense_prompt) in prompts:
        if sparse_prompt is not None:
            point_coord = sparse_prompt["point_coords"].cpu().clone()
            point_label = sparse_prompt["point_labels"].cpu().clone()

            # Visualize sparse prompts (points and optional boxes)
            for m_id in range(point_coord.shape[0]):
                points = point_coord[m_id]
                labels = point_label[m_id]

                points[:, 0] = points[:, 0] * scale_x
                points[:, 1] = points[:, 1] * scale_y

                if 2 in labels:
                    bbox_indices = labels > 1
                    bbox_points = points[bbox_indices]

                    bbox = bbox_points.reshape(1, -1).squeeze()
                    show_box(bbox, prompt_ax)

                point_indices = labels <= 1
                if torch.any(point_indices):
                    point_coords = points[point_indices]
                    point_labels = labels[point_indices]
                    show_points(point_coords, point_labels, prompt_ax)

        if dense_prompt is not None:
            colors = []
            dense_prompt = dense_prompt.squeeze(0)
            for m_id in range(dense_prompt.shape[0]):
                thresholded_prompt = (dense_prompt[m_id].squeeze() > 0.0).to(torch.float)
                thresholded_prompt = F.interpolate(
                    thresholded_prompt[None, None, ...], size=(orig_h, orig_w), mode="bilinear"
                ).squeeze()

                color = show_mask(thresholded_prompt.cpu().numpy(), prompt_ax, color=None)
                colors.append(color)

    # Plot per-object GT, prediction, and error overlays
    for m_id in range(target.shape[0]):
        t = target[m_id]
        p = pred[m_id]
        l = logit[m_id]

        probabilities = torch.sigmoid(l)
        residuals = torch.abs(probabilities - t)
        error = torch.abs(p - t)

        summed_residuals = summed_residuals + residuals.cpu().numpy()

        color = colors[m_id] if colors is not None and len(colors) == target.shape[0] else None
        color = show_mask(t.cpu().numpy(), gt_ax, color=color, opacity=1.0)
        color = show_mask(p.cpu().numpy(), pred_ax, color=color, opacity=1.0)
        color = show_mask(error.cpu().numpy(), err_ax, color=color, opacity=1.0)

    # Aggregate across objects for uncertainty/STD and residuals
    unc_ax.imshow(torch.max(umap, dim=0).values.cpu().numpy(), cmap="magma")
    std_ax.imshow(torch.max(std_map, dim=0).values.cpu().numpy(), cmap="magma")
    res_ax.imshow(summed_residuals, cmap="magma")

    os.makedirs(save_path, exist_ok=True)
    fig_unc.savefig(os.path.join(save_path, "uncertainty.png"), bbox_inches="tight")
    fig_std.savefig(os.path.join(save_path, "std.png"), bbox_inches="tight")
    fig_res.savefig(os.path.join(save_path, "residuals.png"), bbox_inches="tight")
    fig_pred.savefig(os.path.join(save_path, "predicted_masks.png"), bbox_inches="tight")
    fig_err.savefig(os.path.join(save_path, "error_map.png"), bbox_inches="tight")
    fig_prompt.savefig(os.path.join(save_path, "prompt_input.png"), bbox_inches="tight")
    fig_gt.savefig(os.path.join(save_path, "ground_truth_mask.png"), bbox_inches="tight")

    plt.close(fig_unc)
    plt.close(fig_res)
    plt.close(fig_pred)
    plt.close(fig_err)
    plt.close(fig_prompt)
    plt.close(fig_gt)
    plt.close(fig_std)


__all__ = [
    "plot_example",
    "show_box",
    "show_mask",
    "show_masks",
    "show_points",
    "visualise_example",
]
