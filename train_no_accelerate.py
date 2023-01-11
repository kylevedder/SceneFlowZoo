import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from accelerate import Accelerator

from tqdm import tqdm
from dataloaders import ArgoverseSequenceLoader, ArgoverseSequence, SequenceDataset

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


def main_fn():

    def save_checkpoint(batch_idx, model):
        unwrapped_model = model
        state_dict = unwrapped_model.state_dict()
        latest_checkpoint = Path(
            SAVE_FOLDER) / f"checkpoint_{batch_idx:09d}.pt"
        latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, latest_checkpoint)

    # create model and move it to GPU with id rank
    DEVICE = 'cuda'

    sequence_loader = ArgoverseSequenceLoader(
        '/bigdata/argoverse_lidar/train/')
    dataset = SequenceDataset(sequence_loader, SEQUENCE_LENGTH, shuffle=True)

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

    def train_model():
        model.train()
        optimizer.zero_grad()
        subsequence_batch_with_flow = model(
            {k: v.to(DEVICE)
             for k, v in subsequence_batch.items()})
        loss = 0
        loss += loss_fn(subsequence_batch_with_flow, 1)
        loss += loss_fn(subsequence_batch_with_flow, 2)
        writer.add_scalar('loss/train', loss.item(), global_step=batch_idx)
        loss.backward()
        optimizer.step()

    def test_model(train_batch, num_test_steps=20):
        sequence_loader = ArgoverseSequenceLoader(
            '/bigdata/argoverse_lidar/test/')
        dataset = SequenceDataset(sequence_loader, SEQUENCE_LENGTH, shuffle=True)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False)
        model.eval()
        for batch_idx, subsequence_batch in enumerate(dataloader):
            if batch_idx > num_test_steps:
                break
            subsequence_batch_with_flow = model(
                {k: v.to(DEVICE)
                 for k, v in subsequence_batch.items()})
            loss = 0
            loss += loss_fn(subsequence_batch_with_flow, 1)
            loss += loss_fn(subsequence_batch_with_flow, 2)
            print(f"Batch {batch_idx} loss: {loss.item()}")
            writer.add_scalar('loss/test',
                              loss.item(),
                              global_step=train_batch + batch_idx)

    for batch_idx, subsequence_batch in enumerate(tqdm(dataloader)):
        train_model()
        if batch_idx % SAVE_EVERY == 0:
            print(f"Saving checkpoint {batch_idx}")
            save_checkpoint(batch_idx, model)
            test_model(batch_idx)


if __name__ == "__main__":
    main_fn()