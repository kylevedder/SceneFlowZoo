import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from accelerate import Accelerator

from tqdm import tqdm
from dataloaders import ArgoverseSequenceLoader, ArgoverseSequence, SubsequenceDataset

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

accelerator = Accelerator()


def find_latest_checkpoint() -> Path:
    latest_checkpoint = sorted(Path(SAVE_FOLDER).glob("*.pt"))[-1]
    latest_checkpoint = Path(SAVE_FOLDER) / "checkpoint_000012000.pt"
    print("Latest checkpoint", latest_checkpoint)
    return latest_checkpoint


def load_checkpoint(model):
    latest_checkpoint = find_latest_checkpoint()
    model.load_state_dict(torch.load(latest_checkpoint))
    return model


def main_fn():
    # create model and move it to GPU with id rank
    DEVICE = accelerator.device

    sequence_loader = ArgoverseSequenceLoader(
        '/bigdata/argoverse_lidar/train/')
    dataset = SubsequenceDataset(sequence_loader, SEQUENCE_LENGTH)
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
                            NSFP_NUM_LAYERS=NSFP_NUM_LAYERS,
                            zero_prior=True)

    model, dataloader = accelerator.prepare(model, dataloader)

    model = load_checkpoint(model)

    model.eval()
    for batch_idx, subsequence_batch in enumerate(dataloader):
        subsequence_batch_with_flow = model(subsequence_batch)
        delta_1_loss, delta_1_loss_zero_prior = loss_fn(
            subsequence_batch_with_flow, 1, visualize=True)
        delta_2_loss, delta_2_loss_zero_prior = loss_fn(
            subsequence_batch_with_flow, 2)
        loss = delta_1_loss + delta_2_loss
        zero_prior_loss = delta_1_loss_zero_prior + delta_2_loss_zero_prior
        print(
            f"Batch {batch_idx} loss - zero prior: {loss.item() - zero_prior_loss.item()}"
        )


if __name__ == "__main__":
    main_fn()