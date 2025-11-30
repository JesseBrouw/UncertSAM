from typing import Optional

import torch

def prepare_prompt_inputs(
    bbox_prompts: Optional[torch.Tensor],
    masks: torch.Tensor,
    num_points_to_sample:int,
    device:torch.device,
    generator:torch.Generator,
    sampling_strategy:str,
    sample_from_gt_probability: float,
    umaps:Optional[torch.Tensor] = None
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: 
    
    point_coords, point_labels = None, None
    box_coords, box_labels = None, None

    # Input bbox_prompt of Bx1x4
    assert (
        bbox_prompts.ndim == 3
        and bbox_prompts.shape[1] == 1
        and bbox_prompts.shape[-1] == 4
    ), "bbox_prompts must have shape [B, 1, 4]"

    # Default is GT bbox 
    B, _, _ = bbox_prompts.shape
    # Label 2 is special label for top left, and 3 for bottom right.
    box_lab = torch.tensor([2, 3], dtype=torch.int, device=device).repeat(B)

    # reshape to stack of point prompts (top-left and bottom-right corners)
    bbox_prompts = bbox_prompts.reshape(-1, 2, 2)
    box_lab = box_lab.reshape(-1, 2)

    box_coords = bbox_prompts
    box_labels = box_lab

    point_coords, point_labels = sample_point_prompts(masks, generator, device, sampling_strategy, sample_from_gt_probability, num_points_to_sample, umaps)

    return point_coords, point_labels, box_coords, box_labels   # [num_objects, num_points, 2], [num_objects, num_points], [num_objects, num]

def preturb_prompts(
    point_coords: torch.Tensor, point_labels: torch.Tensor, box_coords: torch.Tensor, box_labels: torch.Tensor, target_masks: torch.Tensor, device: torch.device, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
    # Transform back to [num_obj, 4]
    box_coords = box_coords.reshape(-1, 4)
    box_coords = preturb_bounding_box(box_coords, device=device, generator=generator)
    box_coords = box_coords.reshape(-1, 2, 2)
    
    point_coords = preturb_point_prompts(point_coords, device=device, generator=generator)
    
    B, _, _ = point_coords.shape
    xs = point_coords[:, :, 0].long()     # [B, num_points, xcor]
    ys = point_coords[:, :, 1].long()     # [B, num_points, ycor]
    
    l = []
    for b in range(B):
        l.append(target_masks[0, b, ys[b], xs[b]])  # ys[b], xs[b] are [5], get [5] from masks[b]

    point_labels = torch.stack(l).to(device)
    
    return point_coords, point_labels, box_coords, box_labels
     
    
def sample_point_prompts(
    masks: torch.Tensor,
    generator:torch.Generator,
    device:torch.device,
    sampling_strategy:str='uniform',
    sample_from_gt_probability:float=0.5, 
    num_points_to_sample:int=8,
    weight_maps:Optional[torch.Tensor]=None
    ) -> tuple[torch.Tensor, torch.Tensor]: 
    
    output_coords = []
    output_labels = []
    # masks is shape [1 (frames), num_objects, 1024, 1024]
    for obj_idx, m in enumerate(masks.squeeze(0)):
        mb = m.bool()
        H, W = mb.shape
        
        foreground_coords = torch.nonzero(mb, as_tuple=True)
        background_coords = torch.nonzero(~mb, as_tuple=True)
        
        if sampling_strategy == 'uniform':
            fg_indices, bg_indices = set(), set()
            
            point_coords = []
            point_labels = []
            for i in range(num_points_to_sample):
                randnum = 0.0 if i == 0 else torch.rand(1, device=device, generator=generator)  # always sample 1 positive point
                if  randnum < sample_from_gt_probability:
                    index = torch.randint(0, foreground_coords[0].shape[0], size=(1, ), device=device, generator=generator).item()
                    while index in fg_indices:
                        index = torch.randint(0, foreground_coords[0].shape[0], size=(1, ), device=device, generator=generator).item()
                    
                    fg_indices.add(index)
                    point_coords.append(torch.tensor( [ foreground_coords[1][index], foreground_coords[0][index] ] ) )
                    point_labels.append(torch.tensor( [1] ))
                else:
                    index = torch.randint(0, background_coords[0].shape[0], size=(1, ), device=device, generator=generator).item()
                    while index in bg_indices:
                        index = torch.randint(0, background_coords[0].shape[0], size=(1, ), device=device, generator=generator).item()
                    
                    bg_indices.add(index)
                    point_coords.append(torch.tensor( [ background_coords[1][index], background_coords[0][index] ]) )
                    point_labels.append(torch.tensor( [0]))
                    
                    
        elif weight_maps is not None:
            flat_weights = weight_maps[obj_idx].squeeze().flatten()
            flat_gt = m.flatten()
            foreground = (flat_gt == 1)
            background = (flat_gt == 0)
            foreground_weights = flat_weights * foreground   # 0 if background, weight if foreground
            background_weights = flat_weights * background   # 0 if foreground, weight if background
            
            foreground_weights = foreground_weights / foreground_weights.sum()  # Normalize to get probabilities 
            background_weights = background_weights / background_weights.sum()  # Normalize to get probabilities 
            
            point_coords = []
            point_labels = []
            for _ in range(num_points_to_sample):
                if torch.rand(1, device=device, generator=generator) < sample_from_gt_probability:
                    index = torch.multinomial(foreground_weights, 1, replacement=False, device=device, generator=generator).item()
                    foreground_weights[index] = 0.0
                    
                    y = index // W
                    x = index % W
                    point_coords.append(torch.tensor([x, y]))
                    point_labels.append(torch.tensor([1]))
                else:
                    index = torch.multinomial(background_weights, 1, replacement=False, device=device, generator=generator).item()
                    background_weights[index] = 0.0
                    
                    y = index // W
                    x = index % W
                    point_coords.append(torch.tensor([x, y]))
                    point_labels.append(torch.tensor([0]))
                    
        else:
            raise AssertionError("Use uniform sampling, or provide weight maps!")
        
        output_coords.append(torch.stack(point_coords))  # [num_points, 2]
        output_labels.append(torch.cat(point_labels, dim=0)) # [num_points]
    
    return torch.stack(output_coords).to(device), torch.stack(output_labels).to(device)   # [num_objects, num_points, 2], [num_objects, num_points]


def preturb_bounding_box(bbox_prompts: torch.Tensor, device:torch.device, generator:torch.Generator):
    """
    Adds random noise to bounding box coordinates and clamps the results
    to stay within image bounds. Adapted from sam2_utils.py.

    Args:
        bbox_prompts (torch.Tensor): A tensor of bounding box prompts
            with shape `[B, 1, 4]`, where `B` is the batch size.
            Each bounding box is represented by its (x1, y1, x2, y2)
            coordinates.

    Returns:
        torch.Tensor: The perturbed bounding box prompts with the same
            shape as the input: `[B, 1, 4]`.
    """
    
    B, _ = bbox_prompts.shape

    # Get size of boxes
    bbox_w = bbox_prompts[..., 2] - bbox_prompts[..., 0]  # [B, 1]
    bbox_h = bbox_prompts[..., 3] - bbox_prompts[..., 1]  # [B, 1]

    # Default values of bounding box noise for training SAM and SAM 2
    noise_scale = 0.1  # SAM default
    noise_bound = torch.tensor(
        20, device=device
    )  # SAM default

    # Calculate max noise values
    max_x_noise = torch.min(bbox_w * noise_scale, noise_bound)  # [B, 1]
    max_y_noise = torch.min(bbox_h * noise_scale, noise_bound)  # [B, 1]

    # Calculate random noise
    # Generates noise in range [-1, 1] and scales it by max noise bounds
    noise = 2 * torch.rand(B, 4, device=device, generator=generator) - 1 # [B, 1, 4]
    noise = noise * torch.stack(
        (max_x_noise.squeeze(-1), max_y_noise.squeeze(-1), max_x_noise.squeeze(-1), max_y_noise.squeeze(-1)), dim=-1
    ) # [B, 4]

    # Add noise, and clamp such that the boxes remain within image bounds.
    bbox_prompts = bbox_prompts + noise

    img_resolution = 1024
    # Image bounds (0, 0, img_resolution-1, img_resolution-1)
    img_bounds = (torch.tensor(
        [0, 0, img_resolution - 1, img_resolution - 1], device=device
    ))

    # Clamp coordinates to be within image bounds
    bbox_prompts.clamp_(img_bounds.min(), img_bounds.max())

    return bbox_prompts

def preturb_point_prompts(point_prompts: torch.Tensor, device:torch.device, generator:torch.Generator):
    """
    Adds random noise to point prompt coordinates and clamps the results
    to stay within image bounds.

    Args:
        point_prompts (torch.Tensor): A tensor of point prompts
            with shape `[B, N_points, 2]`, where `B` is the batch size
            and `N_points` is the number of point prompts per image.
            Each point is represented by its (x, y) coordinates.

    Returns:
        torch.Tensor: The perturbed point prompts with the same shape
            as the input: `[B, N_points, 2]`.
    """
    B, N_points, _ = point_prompts.shape
    
    # Default values of bounding box noise for training SAM and SAM 2
    noise_bound: torch.Tensor = torch.tensor(
        20, device=device
    )  # SAM default

    # Calculate random noise
    # Generates noise in range [-1, 1] and scales it by noise_bound
    noise = 2 * torch.rand(B, N_points, 2, device=device, generator=generator) - 1 # [B, N_points, 2]
    noise = noise * noise_bound # [B, N_points, 2]

    # Add noise, and clamp such that the points remain within image bounds.
    point_prompts = point_prompts + noise

    img_resolution = 1024
    # Image bounds (0, 0) to (img_resolution-1, img_resolution-1)
    img_bounds_max = torch.tensor([img_resolution - 1, img_resolution - 1], device=device) # [1023, 1023]
    img_bounds_min = torch.tensor([0, 0], device=device) # [0, 0]

    # Clamp coordinates to be within image bounds
    point_prompts.clamp_(img_bounds_min, img_bounds_max)

    return point_prompts