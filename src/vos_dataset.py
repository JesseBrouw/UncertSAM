from __future__ import annotations

import base64
import gc
import json
import logging
import os
import random
from copy import deepcopy
from itertools import accumulate
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from pycocotools import mask as mask_utils
from tensordict import tensorclass

from src.uncertsam2.data import ColorJitter, VideoDatapoint

from external.training.dataset.vos_dataset import (
    MAX_RETRIES,
    VOSDataset as BaseVOSDataset,
)
from external.training.dataset.vos_raw_dataset import VOSRawDataset, VOSFrame, VOSVideo
from external.training.dataset.vos_sampler import (
    EvalSampler as BaseEvalSampler,
    RandomUniformSampler as BaseRandomUniformSampler,
)
from external.training.dataset.vos_segment_loader import (
    LazySegments as BaseLazySegments,
    SA1BSegmentLoader as BaseSA1BSegmentLoader,
)

log = logging.getLogger(__name__)

# Updated SA-1B-style dataset classes to support UncertSAM use cases.
# See https://github.com/facebookresearch/sam2/tree/main/training for original implementations.

@tensorclass
class BatchedVideoMetaData:
    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """Structure returned by ``uncertsam_collate_fn``"""
    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    metadata: BatchedVideoMetaData
    bboxes: torch.FloatTensor
    dict_key: str

    def pin_memory(self, device: Optional[torch.device] = None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        return video_idx * self.num_frames + frame_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        return self.img_batch.transpose(0, 1).flatten(0, 1)


class LazySegments(BaseLazySegments):
    def __getitem__(self, key: int) -> torch.Tensor:  
        if key in self.cache:
            return self.cache[key]
        rle = self.segments[key]
        counts = rle["counts"]
        if isinstance(counts, str):
            counts = base64.b64decode(counts)
        decoded = mask_utils.decode({"size": rle["size"], "counts": counts})
        mask = torch.from_numpy(decoded).clone()
        self.cache[key] = mask
        return mask

    def items(self):
        for key in self.segments.keys():
            yield key, self[key]


class SA1BSegmentLoader(BaseSA1BSegmentLoader):
    """Segment loader backed by an in-memory list of annotations."""

    def __init__(
        self,
        frame_annots: Sequence[dict],
        mask_area_frac_thresh: float = 1.1,
        video_frame_path: Optional[str] = None,
        uncertain_iou: float = -1,
    ) -> None:
        if mask_area_frac_thresh <= 1.0 and video_frame_path is not None:
            with PILImage.open(video_frame_path) as img:
                orig_w, orig_h = img.size
            area = orig_w * orig_h
        else:
            area = None

        rle_masks: List[dict] = []
        mask_ids: List[int] = []
        for frame_annot in frame_annots:
            if frame_annot.get("area", 0) <= 0:
                continue
            if ("uncertain_iou" in frame_annot) and (
                frame_annot["uncertain_iou"] < uncertain_iou
            ):
                continue
            if (
                area is not None
                and mask_area_frac_thresh <= 1.0
                and (frame_annot["area"] / area) >= mask_area_frac_thresh
            ):
                continue
            rle_masks.append(frame_annot["segmentation"])
            mask_ids.append(int(frame_annot["id"]))

        self.segments = LazySegments()
        for i, rle in zip(mask_ids, rle_masks):
            self.segments[i] = rle

    def load(self, frame_idx: int) -> LazySegments:  
        return self.segments


class UncertSAMDataset(VOSRawDataset):
    """Single-frame dataset backed by SA-1B-style JSON annotations."""

    def __init__(
        self,
        dataset_path: str,
        dataset_name: str,
        dataset_split: str,
        mask_area_frac_thresh: float = 1.1,
        uncertain_iou: float = -1,
    ) -> None:
        self.img_folder = os.path.join(dataset_path, dataset_name, dataset_split, "images")
        annotations_file = os.path.join(
            dataset_path, dataset_name, dataset_split, "annotations", "dataset.json"
        )
        with open(annotations_file, "r", encoding="utf-8") as handle:
            annotations = json.load(handle).get("annotations", [])

        self.annotations_df = pd.DataFrame(annotations)
        self.num_frames = 1
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou

        subset = [
            os.path.splitext(name)[0]
            for name in os.listdir(self.img_folder)
            if name.endswith(".jpg")
        ]
        # ``np.string_`` was removed in numpy 2.0; ``np.bytes_`` provides same behavior.
        self.video_names = np.array(sorted(subset)).astype(np.bytes_)

    def get_video(self, idx: int) -> Tuple[VOSVideo, SA1BSegmentLoader]:
        video_name = self.video_names[idx].decode("utf-8")
        image_id = int(video_name.removesuffix(".jpg"))

        video_frame_path = os.path.join(self.img_folder, f"{video_name}.jpg")
        annotations = self.annotations_df[self.annotations_df["image_id"] == image_id]
        annotations_list = annotations.to_dict(orient="records")

        segment_loader = SA1BSegmentLoader(
            annotations_list,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = [VOSFrame(frame_idx, image_path=video_frame_path) for frame_idx in range(self.num_frames)]
        video = VOSVideo(video_name.split("_")[-1], int(video_name), frames)
        return video, segment_loader

    def __len__(self) -> int:
        return len(self.video_names)


class ConcatUncertSAMDataset(UncertSAMDataset):
    """Concatenate multiple ``UncertSAMDataset`` instances with index mapping."""

    def __init__(self, datasets: Sequence[UncertSAMDataset]) -> None:
        self.datasets = list(datasets)
        self.cumulative_sizes = np.array(list(accumulate(len(d) for d in self.datasets)))
        self.mask_area_frac_thresh = self.datasets[0].mask_area_frac_thresh
        self.uncertain_iou = self.datasets[0].uncertain_iou

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def get_video(self, idx: int) -> Tuple[VOSVideo, SA1BSegmentLoader]:
        dataset_idx = next(i for i, size in enumerate(self.cumulative_sizes) if idx < size)
        local_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_video(local_idx)

    def __getitem__(self, idx: int) -> Tuple[VOSVideo, SA1BSegmentLoader]:
        return self.get_video(idx)


class VOSDataset(BaseVOSDataset):
    """Extension of the upstream VOSDataset that adds TTA utilities."""

    def __init__(
        self,
        transforms: Sequence[Callable],
        training: bool,
        video_dataset: VOSRawDataset,
        sampler,
        multiplier: int,
        always_target: bool = True,
        target_segments_available: bool = True,
        tta_transform: Optional[Union[ColorJitter, Sequence[Callable]]] = None,
        tta_samples: int = 5,
    ) -> None:
        super().__init__(
            transforms,
            training,
            video_dataset,
            sampler,
            multiplier,
            always_target,
            target_segments_available,
        )
        self.tta_transform = tta_transform
        self.tta_samples = tta_samples

    def _get_datapoint(self, idx: int):
        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                video, segment_loader = self.video_dataset.get_video(idx)
                sampled = self.sampler.sample(video, segment_loader, epoch=self.curr_epoch)
                break
            except Exception as exc:  # pragma: no cover - mirrors upstream logic
                torch.cuda.empty_cache()
                gc.collect()
                if self.training:
                    logging.warning("Loading failed (id=%s); retry %d due to %s", idx, retry, exc)
                    idx = random.randrange(0, len(self.video_dataset))
                    continue
                raise
        else:
            raise RuntimeError("Unable to fetch datapoint after retries.")

        datapoint = self.construct(video, sampled, segment_loader)
        if self.tta_transform is None:
            return self._apply_eval_transforms(datapoint)
        return self._apply_tta(datapoint)

    def _apply_eval_transforms(self, datapoint: VideoDatapoint):
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def _apply_tta(self, datapoint: VideoDatapoint) -> List[VideoDatapoint]:
        outputs: List[VideoDatapoint] = []
        for _ in range(self.tta_samples):
            dp = deepcopy(datapoint)
            if isinstance(self.tta_transform, ColorJitter):
                dp = self.tta_transform(dp)
                for transform in self._transforms:
                    dp = transform(dp, epoch=self.curr_epoch)
            else:
                for transform in self.tta_transform or []:
                    dp = transform(dp)
            outputs.append(dp)
        gc.collect()
        return outputs


EvalSampler = BaseEvalSampler
RandomUniformSampler = BaseRandomUniformSampler


def uncertsam_collate_fn(
        batch: List[VideoDatapoint],
        dict_key: str,
        laplace: bool,
        out_resolution: int = 128,
        tta: bool = False,
        generator: Optional[torch.Generator] = None,
        **_: Any,
    ) -> Union[BatchedVideoDatapoint, List[BatchedVideoDatapoint]]:
        """
        Args:
            batch: A list of VideoDatapoint instances.
            dict_key (str): A string key used to identify the batch.
        """
        _ = generator  # Unused placeholder; required for compatibility with evaluation pipeline.
        img_batches = []
        
        if tta:
            for b in batch[0]:
                b = [b]
                im_b = []
                for video in b:
                    im_b += [torch.stack([frame.data for frame in video.frames], dim=0)]
                
                im_b = torch.stack(im_b, dim=0).permute((1, 0, 2, 3, 4))  # Batch dim is the second dim 
                img_batches.append(im_b)
        else:
            b = batch
            im_b = []
            for video in batch:
                im_b += [torch.stack([frame.data for frame in video.frames], dim=0)]
                
            im_b = torch.stack(im_b, dim=0).permute((1, 0, 2, 3, 4))  # Batch dim is the second dim 
            img_batches.append(im_b)
        
        img_batch = img_batches[0]
        
        T = img_batch.shape[0]
        # Prepare data structures for sequential processing. Per-frame processing but batched across videos.
        step_t_objects_identifier = [[] for _ in range(T)]
        step_t_frame_orig_size = [[] for _ in range(T)]

        step_t_masks = [[] for _ in range(T)]
        step_t_gt_bboxes = [[] for _ in range(T)]
        
        step_t_obj_to_frame_idx = [
            [] for _ in range(T)
        ]  # List to store frame indices for each time step

        for video_idx, video in enumerate(b):
            orig_video_id = video.video_id
            orig_frame_size = video.size
            for t, frame in enumerate(video.frames):
                objects = frame.objects
                for obj in objects:
                    
                    orig_obj_id = obj.object_id
                    orig_frame_idx = obj.frame_index
                    step_t_obj_to_frame_idx[t].append(
                        torch.tensor([t, video_idx], dtype=torch.int)
                    )
                    
                    obj_segment = obj.segment.to(torch.bool)
                    
                    step_t_masks[t].append(obj_segment.cpu())
                    
                    # Also sample prompts during collate_fn
                    nonzero_coordinates = torch.nonzero(obj_segment, as_tuple=True)
                    y_coordinates, x_coordinates = nonzero_coordinates
                    x_min = x_coordinates.min()
                    y_min = y_coordinates.min()
                    x_max = x_coordinates.max()
                    y_max = y_coordinates.max()
                    
                    step_t_gt_bboxes[t].append(torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float).unsqueeze(0).cpu())
                    
                    step_t_objects_identifier[t].append(
                        torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx]).cpu()
                    )
                    step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size).cpu())

        obj_to_frame_idx = torch.stack(
            [
                torch.stack(obj_to_frame_idx, dim=0)
                for obj_to_frame_idx in step_t_obj_to_frame_idx
            ],
            dim=0,
        )
        masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
        bboxes = torch.stack([torch.stack(bboxes, dim=0) for bboxes in step_t_gt_bboxes], dim=0)
        
        objects_identifier = torch.stack(
            [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
        )
        frame_orig_size = torch.stack(
            [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
        )
        
        output = []
        if laplace:
            output = [(
                    BatchedVideoDatapoint(
                        img_batch=i,
                        obj_to_frame_idx=obj_to_frame_idx,
                        masks=masks,
                        metadata=BatchedVideoMetaData(
                            unique_objects_identifier=objects_identifier,
                            frame_orig_size=frame_orig_size,
                            ),
                        dict_key=dict_key,
                        batch_size=[T],
                        bboxes=bboxes
                        ), F.interpolate(masks.float(), size=(out_resolution, out_resolution), mode='nearest').squeeze(0).reshape(1, -1)
                    ) for i in img_batches]
        else:
            output = [
                BatchedVideoDatapoint(
                    img_batch=i,
                    obj_to_frame_idx=obj_to_frame_idx,
                    masks=masks,
                    metadata=BatchedVideoMetaData(
                        unique_objects_identifier=objects_identifier,
                        frame_orig_size=frame_orig_size,
                    ),
                    dict_key=dict_key,
                    batch_size=[T],
                    bboxes=bboxes
                    )
            for i in img_batches]
        
        
        return output[0] if len(output) == 1 else output


__all__ = [
    "BatchedVideoDatapoint",
    "BatchedVideoMetaData",
    "ConcatUncertSAMDataset",
    "EvalSampler",
    "RandomUniformSampler",
    "SA1BSegmentLoader",
    "UncertSAMDataset",
    "VOSDataset",
    "uncertsam_collate_fn",
]
