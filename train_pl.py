import os

# PL_FAULT_TOLERANT_TRAINING=1
# to enable fault tolerant training
# os.environ['PL_FAULT_TOLERANT_TRAINING'] = '1'

import torch
from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from util_scripts.logging import setup_tb_logger, get_checkpoint_path
from pytorch_lightning.callbacks import ModelCheckpoint

from typing import Optional
import dataloaders
from dataloaders import BucketedSceneFlowDataset, EvalWrapper
from core_utils import ModelWrapper

from pathlib import Path
from mmengine import Config


def make_train_dataloader(cfg: Config) -> tuple[torch.utils.data.DataLoader, EvalWrapper]:
    train_dataset_args = cfg.train_dataset.args
    train_dataset: BucketedSceneFlowDataset = getattr(dataloaders, cfg.train_dataset.name)(
        **train_dataset_args
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, **cfg.train_dataloader.args, collate_fn=train_dataset.collate_fn
    )

    return train_dataloader, train_dataset.evaluator()


def make_val_dataloader(cfg: Config) -> tuple[torch.utils.data.DataLoader, EvalWrapper]:
    test_dataset_args = cfg.test_dataset.args
    test_dataset: BucketedSceneFlowDataset = getattr(dataloaders, cfg.test_dataset.name)(
        **test_dataset_args
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, **cfg.test_dataloader.args, collate_fn=test_dataset.collate_fn
    )

    return test_dataloader, test_dataset.evaluator()


def setup_model(cfg: Config, checkpoint: Optional[Path], evaluator: EvalWrapper):
    if (hasattr(cfg, "is_trainable") and not cfg.is_trainable) or (checkpoint is None):
        model = ModelWrapper(cfg, evaluator=evaluator)
    else:
        assert checkpoint.exists(), f"Checkpoint file {checkpoint} does not exist"
        model = ModelWrapper.load_from_checkpoint(checkpoint, cfg=cfg, evaluator=evaluator)
    return model


def main():
    # Get config file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist"
    cfg = Config.fromfile(args.config)

    if hasattr(cfg, "is_trainable") and not cfg.is_trainable:
        raise ValueError("Config file indicates this model is not trainable.")

    if hasattr(cfg, "seed_everything"):
        pl.seed_everything(cfg.seed_everything)

    checkpoint_path = get_checkpoint_path(cfg)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    # Save config file to checkpoint directory
    cfg.dump(str(checkpoint_path / "config.py"))

    logger = setup_tb_logger(cfg, "train_pl")

    train_dataloader, _ = make_train_dataloader(cfg)
    val_dataloader, evaluator = make_val_dataloader(cfg)

    print("Train dataloader length:", len(train_dataloader))
    print("Val dataloader length:", len(val_dataloader))

    model = setup_model(cfg, args.resume_from_checkpoint, evaluator)

    epoch_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="checkpoint_{epoch:03d}_{step:010d}_epoch_end",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    step_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="checkpoint_{epoch:03d}_{step:010d}",
        save_top_k=-1,
        every_n_train_steps=cfg.save_every,
        save_on_train_epoch_end=True,
    )

    trainer = pl.Trainer(
        devices=args.gpus if not args.cpu else None,
        accelerator="gpu" if not args.cpu else "cpu",
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps=2,
        log_every_n_steps=2,
        val_check_interval=cfg.validate_every,
        check_val_every_n_epoch=(
            cfg.check_val_every_n_epoch if hasattr(cfg, "check_val_every_n_epoch") else 1
        ),
        max_epochs=cfg.epochs,
        accumulate_grad_batches=(
            cfg.accumulate_grad_batches if hasattr(cfg, "accumulate_grad_batches") else 1
        ),
        gradient_clip_val=cfg.gradient_clip_val if hasattr(cfg, "gradient_clip_val") else 0.0,
        callbacks=[epoch_checkpoint_callback, step_checkpoint_callback],
    )
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
