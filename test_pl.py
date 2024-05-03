print("Starting test_pl.py", flush=True)

import torch
from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from core_utils import setup_tb_logger, make_dataloader, setup_model

from pathlib import Path
from mmengine import Config


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

    test_dataloader, evaluator = make_dataloader(
        cfg.test_dataset.name, cfg.test_dataset.args, cfg.test_dataloader.args
    )

    print("Val dataloader length:", len(test_dataloader))

    model = setup_model(cfg, evaluator, args.checkpoint)
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
