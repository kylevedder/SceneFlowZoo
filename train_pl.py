import os
# PL_FAULT_TOLERANT_TRAINING=1
# to enable fault tolerant training
#os.environ['PL_FAULT_TOLERANT_TRAINING'] = '1'

import datetime
import torch
from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from tqdm import tqdm
import dataloaders
from dataloaders import ConcatDataset

from pointclouds import PointCloud, SE3

import models
from model_wrapper import ModelWrapper

from pathlib import Path

try:
    from mmcv import Config
except ImportError:
    from mmengine import Config


def get_rank() -> int:
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_checkpoint_path(cfg, checkpoint_dir_name: str):
    cfg_filename = Path(cfg.filename)
    config_name = cfg_filename.stem
    parent_name = cfg_filename.parent.name
    parent_path = Path(f"model_checkpoints/{parent_name}/{config_name}/")
    rank = get_rank()
    if rank == 0:
        # Since we're rank 0, we can create the directory
        return parent_path / checkpoint_dir_name, checkpoint_dir_name
    else:
        # Since we're not rank 0, we shoulds grab the most recent directory
        checkpoint_path = sorted(parent_path.glob("*"))[-1]
        return checkpoint_path, checkpoint_path.name


def make_train_dataloader(cfg):
    # Handle single loader case
    if not isinstance(cfg.loader, list) and not isinstance(cfg.dataset, list):
        try:
            train_sequence_loader = getattr(dataloaders,
                                            cfg.loader.name)(**cfg.loader.args)
        except Exception as e:
            print("Error loading loader:", cfg.loader.name)
            print("Config:", cfg.loader)
            raise e
        train_dataset = getattr(dataloaders, cfg.dataset.name)(
            sequence_loader=train_sequence_loader, **cfg.dataset.args)
        return torch.utils.data.DataLoader(train_dataset,
                                           **cfg.dataloader.args)

    # Handle multiple loader case
    assert isinstance(cfg.loader, list) and isinstance(cfg.dataset, list), \
        f"Either both loader and dataset should be lists, or neither should be. Got loader: {type(cfg.loader)} and dataset: {type(cfg.dataset)}"

    assert len(cfg.loader) == len(cfg.dataset), \
        f"Length of loader list {len(cfg.loader)} should be equal to length of dataset list {len(cfg.dataset)}"

    print("Using multiple loaders of length:", len(cfg.loader))
    train_sequence_loader_lst = []
    for loader in cfg.loader:
        train_sequence_loader_lst.append(
            getattr(dataloaders, loader.name)(**loader.args))

    print("Using multiple datasets of length:", len(cfg.dataset))
    train_dataloader_lst = []
    for dataset, train_sequence_loader in zip(cfg.dataset,
                                              train_sequence_loader_lst):
        train_dataset = getattr(dataloaders, dataset.name)(
            sequence_loader=train_sequence_loader, **dataset.args)
        train_dataloader_lst.append(train_dataset)

    # Use the concat dataloader to combine the multiple dataloaders
    concat_dataset = dataloaders.ConcatDataset(train_dataloader_lst)

    return torch.utils.data.DataLoader(concat_dataset, **cfg.dataloader.args)


def make_val_dataloader(cfg):
    # Setup val infra
    val_sequence_loader = getattr(dataloaders,
                                  cfg.test_loader.name)(**cfg.test_loader.args)
    val_dataset = getattr(dataloaders, cfg.test_dataset.name)(
        sequence_loader=val_sequence_loader, **cfg.test_dataset.args)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 **cfg.test_dataloader.args)

    return val_dataloader


def setup_model(cfg, checkpoint):
    if hasattr(cfg, "is_trainable") and not cfg.is_trainable:
        model = ModelWrapper(cfg)
    else:
        assert checkpoint is not None, "Must provide checkpoint for validation"
        assert checkpoint.exists(
        ), f"Checkpoint file {checkpoint} does not exist"
        model = ModelWrapper.load_from_checkpoint(checkpoint, cfg=cfg)

    if hasattr(cfg, "compile_pytorch2") and cfg.compile_pytorch2:
        print("PyTorch 2 compile()ing model!")
        model = torch.compile(model, mode="reduce-overhead")
    return model


def main():

    # Get config file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--resume_from_checkpoint', type=Path, default=None)
    parser.add_argument(
        '--checkpoint_dir_name',
        type=str,
        default=datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist"
    cfg = Config.fromfile(args.config)

    if hasattr(cfg, "is_trainable") and not cfg.is_trainable:
        raise ValueError("Config file indicates this model is not trainable.")

    if hasattr(cfg, "seed_everything"):
        pl.seed_everything(cfg.seed_everything)

    checkpoint_path, checkpoint_dir_name = get_checkpoint_path(
        cfg, args.checkpoint_dir_name)

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # Save config file to checkpoint directory
    cfg.dump(str(checkpoint_path / "config.py"))

    tbl = TensorBoardLogger("tb_logs",
                            name=cfg.filename,
                            version=checkpoint_dir_name)

    train_dataloader = make_train_dataloader(cfg)
    val_dataloader = make_val_dataloader(cfg)

    print("Train dataloader length:", len(train_dataloader))
    print("Val dataloader length:", len(val_dataloader))

    resume_from_checkpoint = args.resume_from_checkpoint
    model = setup_model(cfg, resume_from_checkpoint)

    epoch_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="checkpoint_{epoch:03d}_{step:010d}_epoch_end",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True)

    step_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="checkpoint_{epoch:03d}_{step:010d}",
        save_top_k=-1,
        every_n_train_steps=cfg.save_every,
        save_on_train_epoch_end=True)

    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu",
        logger=tbl,
        strategy=DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps=2,
        log_every_n_steps=2,
        val_check_interval=cfg.validate_every,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch if hasattr(
            cfg, "check_val_every_n_epoch") else 1,
        max_epochs=cfg.epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches if hasattr(
            cfg, "accumulate_grad_batches") else 1,
        gradient_clip_val=cfg.gradient_clip_val if hasattr(
            cfg, "gradient_clip_val") else 0.0,
        callbacks=[epoch_checkpoint_callback, step_checkpoint_callback])
    if args.dry_run:
        trainer.validate(model, dataloaders=val_dataloader)
        print("Dry run, exiting")
        exit(0)
    print("Starting training")
    print("Length of train dataloader:", len(train_dataloader))
    print("Length of val dataloader:", len(val_dataloader))
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()