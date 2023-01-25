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

from torch.utils.tensorboard import SummaryWriter

from mmcv import Config

# Get config file from command line
parser = argparse.ArgumentParser()
parser.add_argument('config', type=Path)
args = parser.parse_args()

assert args.config.exists(), f"Config file {args.config} does not exist"
cfg = Config.fromfile(args.config)

writer = SummaryWriter("work_dirs/argoverse/dist_train/", flush_secs=10)

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

    def train():
        model.train()
        for batch_idx, subsequence_batch in enumerate(
                tqdm(dataloader,
                     disable=not accelerator.is_local_main_process,
                     desc=f"Train Epoch {epoch}",
                     leave=True)):
            optimizer.zero_grad()
            model_res = model(subsequence_batch)
            loss = 0
            loss += loss_fn(model_res)
            if accelerator.is_local_main_process:
                writer.add_scalar('loss/train',
                                  loss.item(),
                                  global_step=batch_idx +
                                  len(dataloader) * epoch)
            accelerator.backward(loss)
            optimizer.step()
            if batch_idx % cfg.SAVE_EVERY == 0:
                save_checkpoint(epoch, batch_idx, model)
        save_checkpoint(epoch, None, model)

    def test():
        model.eval()
        with torch.no_grad():
            for batch_idx, subsequence_batch in enumerate(
                    tqdm(test_dataloader,
                         disable=not accelerator.is_local_main_process,
                         desc=f"Test Epoch {epoch}",
                         leave=True)):
                model_res = model(subsequence_batch)
                input_batches = accelerator.gather(subsequence_batch)
                output_batches = accelerator.gather(model_res)
                test_loss_fn.accumulate(batch_idx, input_batches,
                                        output_batches)

            test_loss_fn.finalize(epoch, writer)

    DEVICE = accelerator.device

    # Setup train infra
    sequence_loader = getattr(dataloaders, cfg.loader.name)(**cfg.loader.args)
    dataset = getattr(dataloaders,
                      cfg.dataset.name)(sequence_loader=sequence_loader,
                                        **cfg.dataset.args)
    dataloader = torch.utils.data.DataLoader(dataset, **cfg.dataloader.args)

    loss_fn = getattr(models, cfg.loss_fn.name)(device=DEVICE,
                                                **cfg.loss_fn.args)

    # Setup test infra
    test_sequence_loader = getattr(
        dataloaders, cfg.test_loader.name)(**cfg.test_loader.args)
    test_dataset = getattr(dataloaders, cfg.test_dataset.name)(
        sequence_loader=test_sequence_loader, **cfg.test_dataset.args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  **cfg.test_dataloader.args)
    test_loss_fn = getattr(models,
                           cfg.test_loss_fn.name)(device=DEVICE,
                                                  **cfg.test_loss_fn.args)

    # Setup model
    model = getattr(models, cfg.model.name)(device=DEVICE,
                                            **cfg.model.args).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    model, optimizer, dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, dataloader, test_dataloader)

    for epoch in range(cfg.epochs):
        train()
        test()


if __name__ == "__main__":
    main_fn()