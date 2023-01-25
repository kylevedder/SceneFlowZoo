import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import torchmetrics
import warnings

from tqdm import tqdm
import dataloaders
from dataloaders import ArgoverseSequenceLoader, ArgoverseSequence, SubsequenceDataset, OriginMode

from pointclouds import PointCloud, SE3

import models
from model_wrapper import ModelWrapper

from pathlib import Path

from mmcv import Config

# Get config file from command line
parser = argparse.ArgumentParser()
parser.add_argument('config', type=Path)
args = parser.parse_args()

assert args.config.exists(), f"Config file {args.config} does not exist"
cfg = Config.fromfile(args.config)

test_sequence_loader = getattr(dataloaders,
                               cfg.test_loader.name)(**cfg.test_loader.args)
test_dataset = getattr(dataloaders, cfg.test_dataset.name)(
    sequence_loader=test_sequence_loader, **cfg.test_dataset.args)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              **cfg.test_dataloader.args)
model = ModelWrapper(cfg)
trainer = pl.Trainer(devices=4,
                     accelerator="gpu",
                     strategy="ddp",
                     move_metrics_to_cpu=True)
trainer.validate(model=model, dataloaders=test_dataloader)
