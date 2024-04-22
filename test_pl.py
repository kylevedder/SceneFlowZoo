print("Starting test_pl.py", flush=True)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from bucketed_scene_flow_eval.interfaces import AbstractDataset, AbstractSequenceLoader

from tqdm import tqdm
import datetime
import dataloaders

from core_utils import ModelWrapper

from pathlib import Path

try:
    from mmcv import Config
except ImportError:
    from mmengine import Config


def make_test_dataloader(cfg):
    # There are two supported types of test dataloaders:
    # 1. A dataloader that needs a dependency injected sequence loader
    # 2. A dataloader that needs nothing

    test_dataset_args = cfg.test_dataset.args

    if hasattr(cfg, "test_loader"):
        # Handle case 1
        test_sequence_loader: AbstractSequenceLoader = getattr(dataloaders, cfg.test_loader.name)(
            **cfg.test_loader.args
        )
        test_dataset_args["sequence_loader"] = test_sequence_loader

    test_dataset: AbstractDataset = getattr(dataloaders, cfg.test_dataset.name)(**test_dataset_args)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, **cfg.test_dataloader.args, collate_fn=test_dataset.collate_fn
    )

    return test_dataloader, test_dataset.evaluator()


def setup_model(cfg, evaluator, args):
    if hasattr(cfg, "is_trainable") and not cfg.is_trainable:
        model = ModelWrapper(cfg, evaluator=evaluator)
    else:
        assert args.checkpoint is not None, "Must provide checkpoint for validation"
        assert args.checkpoint.exists(), f"Checkpoint file {args.checkpoint} does not exist"
        model = ModelWrapper.load_from_checkpoint(args.checkpoint, cfg=cfg, evaluator=evaluator)
    return model


def main():
    # Get config file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--checkpoint_dir_name",
        type=str,
        default=datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"),
    )
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist"
    cfg = Config.fromfile(args.config)

    pl.seed_everything(42069)

    tbl = TensorBoardLogger(
        "tb_logs", name=str(Path(cfg.filename).parent.absolute()), version=args.checkpoint_dir_name
    )
    print("Tensorboard logs will be saved to:", tbl.log_dir, flush=True)

    test_dataloader, evaluator = make_test_dataloader(cfg)

    print("Val dataloader length:", len(test_dataloader))

    model = setup_model(cfg, evaluator, args)
    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu" if not args.cpu else "cpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps=2,
        log_every_n_steps=2,
        logger=tbl,
    )
    trainer.validate(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
