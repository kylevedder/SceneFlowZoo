import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse

from accelerate import Accelerator

from tqdm import tqdm
import dataloaders
from dataloaders import ArgoverseSequenceLoader, ArgoverseSequence, SubsequenceDataset, OriginMode

from pointclouds import PointCloud, SE3

import models

from pathlib import Path

from mmcv import Config

# Get config file from command line
parser = argparse.ArgumentParser()
parser.add_argument('config', type=Path)
args = parser.parse_args()

assert args.config.exists(), f"Config file {args.config} does not exist"
cfg = Config.fromfile(args.config)

accelerator = Accelerator()


def main_fn():

    def save_checkpoint(epoch, batch_idx, model):
        if not accelerator.is_local_main_process:
            return
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = unwrapped_model.state_dict()

        checkpoint_folder = Path(cfg.SAVE_FOLDER)
        if batch_idx is not None:
            latest_checkpoint = checkpoint_folder / f"checkpoint_epoch_{epoch:06d}_batch_{batch_idx:09d}.pt"
        else:
            latest_checkpoint = checkpoint_folder / f"checkpoint_epoch_{epoch:06d}_final.pt"
        latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        accelerator.save(state_dict, latest_checkpoint)

    DEVICE = accelerator.device

    test_sequence_loader = getattr(
        dataloaders, cfg.test_loader.name)(**cfg.test_loader.args)
    test_dataset = getattr(dataloaders, cfg.test_dataset.name)(
        sequence_loader=test_sequence_loader, **cfg.test_dataset.args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  **cfg.test_dataloader.args)

    model = getattr(models, cfg.model.name)(device=DEVICE,
                                            **cfg.model.args).to(DEVICE)
    test_loss_fn = getattr(models,
                           cfg.test_loss_fn.name)(device=DEVICE,
                                                  **cfg.test_loss_fn.args)

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    with torch.no_grad():
        model.eval()
        for batch_idx, subsequence_batch in enumerate(
                tqdm(test_dataloader,
                     disable=not accelerator.is_local_main_process,
                     leave=True)):
            model_res = model(subsequence_batch)
            input_batches, output_batches = accelerator.gather_for_metrics(
                (subsequence_batch, model_res))
            # accelerator.print("Before accumulate")
            if accelerator.is_local_main_process:
                test_loss_fn.accumulate(batch_idx, input_batches,
                                        output_batches)
            # accelerator.print("After accumulate")

        if accelerator.is_local_main_process:
            test_loss_fn.finalize()


if __name__ == "__main__":
    main_fn()