import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from accelerate import Accelerator

from tqdm import tqdm
from dataloaders import ArgoverseSequenceLoader, ArgoverseSequence, SubsequenceDataset, OriginMode

from pointclouds import PointCloud, SE3

from model.embedders import Embedder
from model.backbones import FeaturePyramidNetwork
from model.attention import JointConvAttention
from model.heads import NeuralSceneFlowPrior
from model import JointFlow, JointFlowLoss
from pytorch3d.ops.knn import knn_points

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("work_dirs/argoverse/dist_train/", flush_secs=10)

from config_params import *

accelerator = Accelerator()


def main_fn():

    def save_checkpoint(batch_idx, model):
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = unwrapped_model.state_dict()
        latest_checkpoint = Path(SAVE_FOLDER) / f"checkpoint_{batch_idx:09d}.pt"
        latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        accelerator.save(
            state_dict,
            latest_checkpoint
        )

    # create model and move it to GPU with id rank
    DEVICE = accelerator.device

    sequence_loader = ArgoverseSequenceLoader(
        '/bigdata/argoverse_lidar/train/')
    dataset = SubsequenceDataset(sequence_loader, SEQUENCE_LENGTH, OriginMode.FIRST_ENTRY)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)

    model = JointFlow(BATCH_SIZE, DEVICE, VOXEL_SIZE, PSEUDO_IMAGE_DIMS,
                      POINT_CLOUD_RANGE, MAX_POINTS_PER_VOXEL,
                      FEATURE_CHANNELS, FILTERS_PER_BLOCK, PYRAMID_LAYERS,
                      SEQUENCE_LENGTH, NSFP_FILTER_SIZE,
                      NSFP_NUM_LAYERS).to(DEVICE)
    loss_fn = JointFlowLoss(device=DEVICE,
                            NSFP_FILTER_SIZE=NSFP_FILTER_SIZE,
                            NSFP_NUM_LAYERS=NSFP_NUM_LAYERS)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer,
                                                       dataloader)

    for batch_idx, subsequence_batch in enumerate(
            tqdm(dataloader, disable=not accelerator.is_local_main_process)):
        optimizer.zero_grad()
        subsequence_batch_with_flow = model(subsequence_batch)
        loss = 0
        loss += loss_fn(subsequence_batch_with_flow, 1)
        loss += loss_fn(subsequence_batch_with_flow, 2)
        if accelerator.is_local_main_process:
            writer.add_scalar('loss/train', loss.item(), global_step=batch_idx)
        accelerator.backward(loss)
        optimizer.step()
        if batch_idx % SAVE_EVERY == 0 and accelerator.is_local_main_process:
            save_checkpoint(batch_idx, model)


if __name__ == "__main__":
    main_fn()