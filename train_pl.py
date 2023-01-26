import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DDPStrategy
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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

tbl = TensorBoardLogger("tb_logs", name="train")

# Setup train infra
train_sequence_loader = getattr(dataloaders,
                                cfg.loader.name)(**cfg.loader.args)
train_dataset = getattr(dataloaders, cfg.dataset.name)(
    sequence_loader=train_sequence_loader, **cfg.dataset.args)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               **cfg.dataloader.args)

val_sequence_loader = getattr(dataloaders,
                              cfg.test_loader.name)(**cfg.test_loader.args)
val_dataset = getattr(dataloaders, cfg.test_dataset.name)(
    sequence_loader=val_sequence_loader, **cfg.test_dataset.args)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             **cfg.test_dataloader.args)
model = ModelWrapper(cfg)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"model_checkpoints/{cfg.model.name}/",
    filename="checkpoint_{epoch:03d}_{step:010d}",
    save_top_k=-1,
    every_n_train_steps=cfg.save_every)

trainer = pl.Trainer(devices=4,
                     accelerator="gpu",
                     logger=tbl,
                     strategy=DDPStrategy(find_unused_parameters=False),
                     move_metrics_to_cpu=False,
                     num_sanity_val_steps=-1,
                     log_every_n_steps=2,
                     val_check_interval=cfg.validate_every,
                     callbacks=[checkpoint_callback])
trainer.fit(model, train_dataloader, val_dataloader)
