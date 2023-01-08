import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm
from dataloaders import ArgoverseSequenceLoader, ArgoverseSequence, SequenceDataset

from pointclouds import PointCloud, SE3

from model.embedders import Embedder
from model.backbones import FeaturePyramidNetwork
from model.attention import JointConvAttention
from model.heads import NeuralSceneFlowPrior
from model import JointFlow, JointFlowLoss
from pytorch3d.ops.knn import knn_points

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("work_dirs/argoverse/dist_train/", flush_secs=10)

BATCH_SIZE = 1
VOXEL_SIZE = (0.14, 0.14, 4)
PSEUDO_IMAGE_DIMS = (512, 512)
POINT_CLOUD_RANGE = (-33.28, -33.28, -3, 33.28, 33.28, 1)
MAX_POINTS_PER_VOXEL = 128
FEATURE_CHANNELS = 16
FILTERS_PER_BLOCK = 3
PYRAMID_LAYERS = 1

SEQUENCE_LENGTH = 6

NSFP_FILTER_SIZE = 64
NSFP_NUM_LAYERS = 4

LEARNING_RATE = 1e-4
GRADIENT_MAGNITUDE_CLIP = 35

ITERATIONS_PER_SUBSEQUENCE = 2


def main_fn():
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        DDP = DistributedDataParallel
        print(f"Running on rank {rank} with batch size {BATCH_SIZE}.")
    except ValueError:
        rank = 0
        DDP = lambda x, device_ids: x.to(device_ids[0])
        print(f"Running on single GPU with batch size {BATCH_SIZE}.")

    def save_checkpoint(batch_idx, model, optimizer, scheduler, loss):
        torch.save(
            {
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
            }, f"work_dirs/argoverse/dist_train/checkpoint_{batch_idx:09d}.pt")

    # create model and move it to GPU with id rank
    DEVICE = rank % torch.cuda.device_count()

    sequence_loader = ArgoverseSequenceLoader(
        '/bigdata/argoverse_lidar/train/')
    dataset = SequenceDataset(sequence_loader, SEQUENCE_LENGTH)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)

    model = JointFlow(BATCH_SIZE, DEVICE, VOXEL_SIZE, PSEUDO_IMAGE_DIMS,
                      POINT_CLOUD_RANGE, MAX_POINTS_PER_VOXEL,
                      FEATURE_CHANNELS, FILTERS_PER_BLOCK, PYRAMID_LAYERS,
                      SEQUENCE_LENGTH, NSFP_FILTER_SIZE,
                      NSFP_NUM_LAYERS).to(DEVICE)
    model = DDP(model, device_ids=[DEVICE])
    loss_fn = JointFlowLoss(device=DEVICE,
                            NSFP_FILTER_SIZE=NSFP_FILTER_SIZE,
                            NSFP_NUM_LAYERS=NSFP_NUM_LAYERS)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=LEARNING_RATE,
        max_lr=LEARNING_RATE * 10,
        step_size_up=len(dataloader) // 2,
        cycle_momentum=False)

    for batch_idx, subsequence_batch in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        subsequence_batch_with_flow = model(subsequence_batch)
        loss = 0
        loss += loss_fn(subsequence_batch_with_flow, 1)
        loss += loss_fn(subsequence_batch_with_flow, 2)
        writer.add_scalar('loss/train', loss.item(), global_step=batch_idx)
        writer.add_scalar('lr',
                            scheduler.get_last_lr()[0],
                            global_step=batch_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                        max_norm=GRADIENT_MAGNITUDE_CLIP,
                                        norm_type=2)
        optimizer.step()
        scheduler.step()
        if batch_idx % 100 == 0:
            save_checkpoint(batch_idx, model, optimizer, scheduler, loss)


if __name__ == "__main__":
    main_fn()