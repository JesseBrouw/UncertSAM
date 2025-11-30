from __future__ import annotations

import logging
import time
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.uncertainsam2 import UncertainSAM2
from src.uncertsam2.config import build_uncertain_sam2_kwargs
from src.uncertsam2.data import ColorJitter
from src.uncertsam2.evaluation import evaluate_dataset
from src.utils import get_sam, prepare_modules_for_finetuning, setup_device
from src.vos_dataset import EvalSampler, UncertSAMDataset, VOSDataset, uncertsam_collate_fn

hydra.core.global_hydra.GlobalHydra.instance().clear()
log = logging.getLogger(__name__)
device = setup_device()


@hydra.main(version_base=None, config_path="../../configs", config_name="local_config")
def main(cfg: Any) -> None:
    """Entry point for Hydra-driven evaluation runs."""
    log_dir = HydraConfig.get().runtime.output_dir
    log.info(log_dir)
    
    checkpoint_model = get_sam(
        cfg.experiment.sam_model,
        cfg.checkpoint_mapping,
        device=device,
        train_mode=False,
        multimask_mode=False,
    )

    laplace_target_module = None
    laplace_checkpoint = None

    if cfg.experiment.laplace_checkpoint is not None:
        laplace_checkpoint = torch.load(cfg.experiment.laplace_checkpoint, map_location=device)
        laplace_target_module = "sam_mask_decoder.output_hypernetworks_mlps.0.layers.2"
        log.info("Loaded Laplace checkpoint for %s", laplace_target_module)

    if cfg.experiment.variance_network_checkpoint is not None:
        targets = ["model.sam_mask_decoder."]
        variance_network_ckpt = torch.load(
            cfg.experiment.variance_network_checkpoint,
            map_location=device,
            weights_only=False,
        )
        state_dict = variance_network_ckpt["state_dict"]

        uncertainty_statedict = OrderedDict()
        for key, value in state_dict.items():
            for target in targets:
                if key.startswith(target):
                    uncertainty_statedict[key.replace(target, "")] = value

        checkpoint_model.sam_mask_decoder.load_state_dict(uncertainty_statedict, strict=False)

    if laplace_target_module is not None:
        checkpoint_model = prepare_modules_for_finetuning(
            checkpoint_model, [laplace_target_module]
        )

    uncertain_kwargs = build_uncertain_sam2_kwargs(
        sam_base_model=checkpoint_model,
        experiment_cfg=cfg.experiment,
        device=device,
        laplace_checkpoint=laplace_checkpoint,
        laplace_target_module=laplace_target_module,
    )
    model = UncertainSAM2(**uncertain_kwargs)

    target_datasets = defaultdict()
    for d in cfg.experiment.target_datasets:
        target_datasets[d] = UncertSAMDataset(
            dataset_path=cfg.data_dir,
            dataset_name=d,
            dataset_split=cfg.experiment.split,
        )

    sampler = EvalSampler()
    transforms = instantiate(cfg.experiment.eval_transforms)

    tta_transform = cfg.experiment.tta_transform
    tta_transform = tta_transform if cfg.experiment.uncertainty_method == "TTA" else None
    match tta_transform:
        case "hue":
            tta_transform = ColorJitter(
                consistent_transform=True,
                brightness=0.0,
                contrast=0.0,
                saturation=0.0,
                hue=0.5,
            )
        case "in_domain":
            tta_transform = instantiate(cfg.experiment.in_domain_tta_transform)
        case _:
            tta_transform = None

    log.info("TTA transform : %s", tta_transform)

    vos_datasets = defaultdict()
    for key, value in target_datasets.items():
        vos_datasets[key] = VOSDataset(
            transforms=transforms,
            training=True,
            video_dataset=value,
            sampler=sampler,
            multiplier=1,
            tta_transform=tta_transform,
            tta_samples=cfg.experiment.tta_samples,
        )

    dataloaders = {}
    generators = {}
    for key, value in vos_datasets.items():
        generators[key] = torch.Generator().manual_seed(cfg.experiment.seed)
        custom_collate = partial(
            uncertsam_collate_fn,
            dict_key="test",
            laplace=False,
            generator=generators[key],
            out_resolution=128,
            tta=(tta_transform is not None),
        )
        dataloaders[key] = DataLoader(
            value,
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )

    start = time.time()
    for key, value in dataloaders.items():
        log.info("----- Starting experiment for %s -----", key)
        generator = torch.Generator(device=device).manual_seed(cfg.experiment.seed)
        evaluate_dataset(
            value,
            model,
            key,
            cfg.data_dir,
            target_datasets[key].annotations_df,
            generator=generator,
            device=device,
            log_dir=log_dir,
            split=cfg.experiment.split,
        )

    log.info("Done! Evaluation took %.2f seconds.", time.time() - start)


if __name__ == "__main__":
    main()
