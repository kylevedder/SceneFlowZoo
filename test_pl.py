import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torchmetrics
import warnings

from tqdm import tqdm
import dataloaders
from dataloaders import ArgoverseRawSequenceLoader, ArgoverseRawSequence, SubsequenceRawDataset, OriginMode

from pointclouds import PointCloud, SE3

import models
from model_wrapper import ModelWrapper

from pathlib import Path

try:
    from mmcv import Config
except ImportError:
    from mmengine import Config

# Get config file from command line
parser = argparse.ArgumentParser()
parser.add_argument('config', type=Path)
parser.add_argument('--checkpoint', type=Path, default=None)
parser.add_argument('--gpus', type=int, default=1)
args = parser.parse_args()

assert args.config.exists(), f"Config file {args.config} does not exist"
cfg = Config.fromfile(args.config)


def make_test_dataloader(cfg):
    # Setup val infra
    test_sequence_loader = getattr(
        dataloaders, cfg.test_loader.name)(**cfg.test_loader.args)
    test_dataset = getattr(dataloaders, cfg.test_dataset.name)(
        sequence_loader=test_sequence_loader, **cfg.test_dataset.args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  **cfg.test_dataloader.args)

    return test_dataloader


def setup_model(cfg):
    if hasattr(cfg, "is_trainable") and not cfg.is_trainable:
        model = ModelWrapper(cfg)
    else:
        assert args.checkpoint is not None, "Must provide checkpoint for validation"
        assert args.checkpoint.exists(
        ), f"Checkpoint file {args.checkpoint} does not exist"
        model = ModelWrapper.load_from_checkpoint(args.checkpoint, cfg=cfg)

    
    if hasattr(cfg, "compile_pytorch2") and cfg.compile_pytorch2:
        print("PyTorch 2 compile()ing model!")
        model = torch.compile(model, mode="reduce-overhead")
    return model


pl.seed_everything(42069)

test_dataloader = make_test_dataloader(cfg)

print("Val dataloader length:", len(test_dataloader))

model = setup_model(cfg)
trainer = pl.Trainer(devices=args.gpus,
                         accelerator="gpu",
                         strategy=DDPStrategy(find_unused_parameters=False),
                         num_sanity_val_steps=2,
                         log_every_n_steps=2,
                         val_check_interval=cfg.validate_every,
                         max_epochs=cfg.epochs,
                         accumulate_grad_batches=cfg.accumulate_grad_batches
                         if hasattr(cfg, "accumulate_grad_batches") else 1,
                         gradient_clip_val=cfg.gradient_clip_val if hasattr(
                             cfg, "gradient_clip_val") else 0.0)
trainer.validate(model, dataloaders=test_dataloader)
