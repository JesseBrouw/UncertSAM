import logging

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Updated build sam 2 function in incorporate uncertainty features.
# See https://github.com/facebookresearch/sam2 for original source code and license.


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    train_mode=False,
    multimask_mode=True,
    **kwargs,
):

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg.model, False)
    cfg.model._target_ = 'src.uncertsam2.modeling.sam2_base.SAM2Base'
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)

    model = model.to(device)
    
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

        if missing_keys:
            logging.warning(f"Missing keys (likely new layers): {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys (ignored): {unexpected_keys}")
            
        logging.info("Loaded checkpoint sucessfully")
