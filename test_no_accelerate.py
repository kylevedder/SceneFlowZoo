import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

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

from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from config_params import *


def find_latest_checkpoint() -> Path:
    latest_checkpoint = sorted(Path(SAVE_FOLDER).glob("*.pt"))[-1]
    print("Latest checkpoint", latest_checkpoint)
    return latest_checkpoint


def load_checkpoint(model):
    latest_checkpoint = find_latest_checkpoint()
    model.load_state_dict(torch.load(latest_checkpoint))
    return model


def main_fn():
    # create model and move it to GPU with id rank
    DEVICE = 'cuda'

    sequence_loader = ArgoverseSequenceLoader(
        '/bigdata/argoverse_lidar/train/')
    dataset = SequenceDataset(sequence_loader, SEQUENCE_LENGTH)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)

    model = JointFlow(BATCH_SIZE, DEVICE, VOXEL_SIZE, PSEUDO_IMAGE_DIMS,
                      POINT_CLOUD_RANGE, MAX_POINTS_PER_VOXEL,
                      FEATURE_CHANNELS, FILTERS_PER_BLOCK, PYRAMID_LAYERS,
                      SEQUENCE_LENGTH, NSFP_FILTER_SIZE, NSFP_NUM_LAYERS)
    loss_fn = JointFlowLoss(device=DEVICE,
                            NSFP_FILTER_SIZE=NSFP_FILTER_SIZE,
                            NSFP_NUM_LAYERS=NSFP_NUM_LAYERS)

    model = load_checkpoint(model)
    model.to(DEVICE)

    model.eval()
    for batch_idx, subsequence_batch in enumerate(dataloader):
        subsequence_batch_with_flow = model(
            {k: v.to(DEVICE)
             for k, v in subsequence_batch.items()})
        loss = 0
        loss += loss_fn(subsequence_batch_with_flow, 1)
        loss += loss_fn(subsequence_batch_with_flow, 2)
        print(f"Batch {batch_idx} loss: {loss.item()}")


if __name__ == "__main__":
    main_fn()