import logging
import os
import random
from typing import Dict, List
import numpy as np 

import torch

from external.segment_anything.build_sam import sam_model_registry
from src.uncertsam2.build_sam import build_sam2
from src.uncertsam2.visualization import (
    show_box,
    show_mask,
    show_masks,
    show_points,
    visualise_example,
)

log = logging.getLogger(__name__)

def get_sam(sam_model:str, checkpoint_mapping:Dict, device:torch.device, train_mode:bool=False, multimask_mode:bool=True) -> torch.nn.Module: 
    target_SAM = sam_model
    ckpt_info = checkpoint_mapping[target_SAM]

    if '2' in target_SAM:
        model = build_sam2(config_file=ckpt_info[0], ckpt_path=ckpt_info[1], device=device, mode='eval', train_mode=train_mode, multimask_mode=multimask_mode)
        model = model.to(device)
    else:
        log.info(ckpt_info)
        model = sam_model_registry[ckpt_info[0]](checkpoint=os.getcwd() + ckpt_info[1])
        model = model.to(device)

    log.info(f'Successfully loaded {target_SAM}')
    
    return model

def setup_device() -> torch.device: 
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}.")
    return device

def set_seed(seed: int) -> None: 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def prepare_modules_for_finetuning(model: torch.nn.Module, parameter_subset:List[str]) -> torch.nn.Module: 
    """ Select specific modules of the network, and set those parts to trainable. """
    if hasattr(model, 'sam2_train_model'): 
        model = model.sam2_train_model
        
    for name, module in model.named_modules():
        if any(map(lambda x: x in str(name), parameter_subset)):
            print(name)
            for param in module.parameters():
                param.requires_grad = True 
        else:
            for param in module.parameters():
                param.requires_grad = False 
            
    return model 
