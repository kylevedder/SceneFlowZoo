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

ITERATIONS_PER_SUBSEQUENCE = 2

accelerator = Accelerator(mixed_precision="fp16")


def main_fn():

    def save_checkpoint(batch_idx, model, optimizer, loss):
        torch.save(
            {
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"/bigdata/offline_sceneflow_checkpoints/checkpoint_{batch_idx:09d}.pt")

    # create model and move it to GPU with id rank
    DEVICE = accelerator.device

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
    loss_fn = JointFlowLoss(device=DEVICE,
                            NSFP_FILTER_SIZE=NSFP_FILTER_SIZE,
                            NSFP_NUM_LAYERS=NSFP_NUM_LAYERS)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer,
                                                       dataloader)

    for batch_idx, subsequence_batch in enumerate(
            tqdm(dataloader, disable=not accelerator.is_local_main_process)):
        optimizer.zero_grad()
        with accelerator.autocast():
            subsequence_batch_with_flow = model(subsequence_batch)
        loss = 0
        loss += loss_fn(subsequence_batch_with_flow, 1)
        loss += loss_fn(subsequence_batch_with_flow, 2)
        if accelerator.is_local_main_process:
            writer.add_scalar('loss/train', loss.item(), global_step=batch_idx)
        accelerator.backward(loss)
        optimizer.step()
        if batch_idx % 500 == 0 and accelerator.is_local_main_process:
            save_checkpoint(batch_idx, model, optimizer, loss)


if __name__ == "__main__":
    main_fn()