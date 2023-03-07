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

from mmcv import Config

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
    return model


test_dataloader = make_test_dataloader(cfg)
model = setup_model(cfg)

trainer = pl.Trainer(
    devices=args.gpus,
    accelerator="gpu",
    strategy=DDPStrategy(find_unused_parameters=False),
    resume_from_checkpoint=args.checkpoint,
    move_metrics_to_cpu=False,
)
trainer.validate(model=model, dataloaders=test_dataloader)
