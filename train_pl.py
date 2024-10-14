import os

# PL_FAULT_TOLERANT_TRAINING=1
# to enable fault tolerant training
# os.environ['PL_FAULT_TOLERANT_TRAINING'] = '1'

import torch
from pathlib import Path
import argparse
from core_utils import setup_tb_logger, get_checkpoint_path, make_dataloader, setup_model
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint


from pathlib import Path
from mmengine import Config


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

    train_dataloader, _ = make_dataloader(
        cfg.train_dataset.name, cfg.train_dataset.args, cfg.train_dataloader.args
    )
    val_dataloader, evaluator = make_dataloader(
        cfg.test_dataset.name, cfg.test_dataset.args, cfg.test_dataloader.args
    )

    print("Train dataloader length:", len(train_dataloader))
    print("Val dataloader length:", len(val_dataloader))

    model = setup_model(cfg, evaluator, args.resume_from_checkpoint)

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
