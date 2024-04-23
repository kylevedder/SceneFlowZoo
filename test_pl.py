print("Starting test_pl.py", flush=True)

import torch
from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from util_scripts.logging import setup_tb_logger
from dataloaders import EvalWrapper, BucketedSceneFlowDataset

import dataloaders

from core_utils import ModelWrapper

from pathlib import Path
from mmengine import Config


def make_test_dataloader(cfg: Config) -> tuple[torch.utils.data.DataLoader, EvalWrapper]:
    test_dataset_args = cfg.test_dataset.args
    test_dataset: BucketedSceneFlowDataset = getattr(dataloaders, cfg.test_dataset.name)(
        **test_dataset_args
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, **cfg.test_dataloader.args, collate_fn=test_dataset.collate_fn
    )

    return test_dataloader, test_dataset.evaluator()


def setup_model(cfg: Config, evaluator: EvalWrapper, args):
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
    args = parser.parse_args()

    assert args.config.exists(), f"Config file {args.config} does not exist"
    cfg = Config.fromfile(args.config)

    pl.seed_everything(42069)
    logger = setup_tb_logger(cfg, "test_pl")

    test_dataloader, evaluator = make_test_dataloader(cfg)

    print("Val dataloader length:", len(test_dataloader))

    model = setup_model(cfg, evaluator, args)
    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator="gpu" if not args.cpu else "cpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        num_sanity_val_steps=2,
        log_every_n_steps=2,
        logger=logger,
    )
    trainer.validate(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
