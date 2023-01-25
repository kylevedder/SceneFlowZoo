import torch
import torch.nn.functional as F
from dataloaders import PointCloudDataset, ArgoverseSequenceLoader
from models import PretrainEmbedding
from tqdm import tqdm

from configs.first_attempt.config import *

from accelerate import Accelerator

from pathlib import Path

accelerator = Accelerator()
DEVICE = accelerator.device
BATCH_SIZE = 32

LEARNING_RATE = 1e-4
PRETRAIN_SAVE_FOLDER = "/efs/embedding_pretrain_saves/"

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("work_dirs/argoverse/pretrain/", flush_secs=5)
sequence_loader = ArgoverseSequenceLoader('/efs/argoverse_lidar/train/')
dataset = PointCloudDataset(sequence_loader)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)

model = PretrainEmbedding(BATCH_SIZE, DEVICE, VOXEL_SIZE, PSEUDO_IMAGE_DIMS,
                          POINT_CLOUD_RANGE, MAX_POINTS_PER_VOXEL,
                          FEATURE_CHANNELS, FILTERS_PER_BLOCK, PYRAMID_LAYERS)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def save_checkpoint(batch_idx, model):
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = unwrapped_model.state_dict()
    latest_checkpoint = Path(
        PRETRAIN_SAVE_FOLDER) / f"checkpoint_{batch_idx:09d}.pt"
    latest_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    accelerator.save(state_dict, latest_checkpoint)


def pretrain_loss(tensor_out, scales, shears, is_uppers):
    loss = 0
    for scale, shear, is_upper, tensor in zip(scales, shears, is_uppers,
                                              tensor_out):
        # L1 norm on scale
        loss += torch.linalg.norm(tensor[0] - scale, ord=1)
        # L1 norm on shear
        loss += torch.linalg.norm(tensor[1] - shear, ord=1)
        # Cross entropy on is_upper
        loss += F.binary_cross_entropy(torch.sigmoid(tensor[2]),
                                       is_upper.reshape(-1, 1).float())
    return loss


model, optimizer, dataloader = accelerator.prepare(model, optimizer,
                                                   dataloader)

for batch_idx, point_cloud_batch in enumerate(
        tqdm(dataloader, disable=not accelerator.is_local_main_process)):
    optimizer.zero_grad()

    pcs = point_cloud_batch['pc'].float()
    translations = point_cloud_batch['translation'].float()
    scales = point_cloud_batch['random_scale'].float()
    shears = point_cloud_batch['shear_amount'].float()
    is_uppers = point_cloud_batch['is_upper']

    tensor_out = model(pcs, translations)
    loss = pretrain_loss(tensor_out, scales, shears, is_uppers)

    if accelerator.is_local_main_process:
        writer.add_scalar('loss/train', loss.item(), global_step=batch_idx)
    accelerator.backward(loss)
    optimizer.step()
    if batch_idx % save_every_iter == 0 and accelerator.is_local_main_process:
        save_checkpoint(batch_idx, model)

print("Save final model...")
if accelerator.is_local_main_process:
    save_checkpoint(batch_idx, model)
