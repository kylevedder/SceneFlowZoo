import datetime
import torch
from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only, rank_zero
from pytorch_lightning.strategies import DDPStrategy
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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
parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
parser.add_argument('--resume_from_checkpoint', type=Path, default=None)
args = parser.parse_args()

assert args.config.exists(), f"Config file {args.config} does not exist"
cfg = Config.fromfile(args.config)

if hasattr(cfg, "is_trainable") and not cfg.is_trainable:
    raise ValueError("Config file indicates this model is not trainable.")

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

if args.resume_from_checkpoint is not None:
    assert args.resume_from_checkpoint.exists(
    ), f"Checkpoint file {args.resume_from_checkpoint} does not exist"
    model = ModelWrapper.load_from_checkpoint(args.resume_from_checkpoint,
                                              cfg=cfg)
else:
    model = ModelWrapper(cfg)


def setup_checkpoint_dir(cfg):
    cfg_filename = Path(cfg.filename)
    config_name = cfg_filename.stem
    parent_name = cfg_filename.parent.name
    time_str = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    checkpoint_path = Path(
        f"model_checkpoints/{parent_name}/{config_name}/{time_str}")
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # Save config file to checkpoint directory
    cfg.dump(checkpoint_path / "config.py")
    return checkpoint_path


checkpoint_callback = ModelCheckpoint(
    dirpath=setup_checkpoint_dir(cfg),
    filename="checkpoint_{epoch:03d}_{step:010d}",
    save_top_k=-1,
    every_n_train_steps=cfg.save_every,
    save_on_train_epoch_end=True)

trainer = pl.Trainer(devices=args.gpus,
                     accelerator="gpu",
                     logger=tbl,
                     strategy=DDPStrategy(find_unused_parameters=False),
                     move_metrics_to_cpu=False,
                     num_sanity_val_steps=2,
                     log_every_n_steps=2,
                     val_check_interval=cfg.validate_every,
                     max_epochs=cfg.epochs,
                     accumulate_grad_batches=cfg.accumulate_grad_batches
                     if hasattr(cfg, "accumulate_grad_batches") else 1,
                     gradient_clip_val=cfg.gradient_clip_val if hasattr(
                         cfg, "gradient_clip_val") else 0.0,
                     callbacks=[checkpoint_callback])
trainer.fit(model, train_dataloader, val_dataloader)
