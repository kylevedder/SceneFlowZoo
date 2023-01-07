import torch
from tqdm import tqdm
from dataloaders import ArgoverseSequenceLoader, ArgoverseSequence, SequenceDataset

from pointclouds import PointCloud, SE3

from model.embedders import Embedder
from model.backbones import FeaturePyramidNetwork
from model.attention import JointConvAttention
from model.heads import NeuralSceneFlowPrior
from model import JointFlow
from pytorch3d.ops.knn import knn_points

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("work_dirs/argoverse/")

VOXEL_SIZE = (0.14, 0.14, 4)
PSEUDO_IMAGE_DIMS = (512, 512)
POINT_CLOUD_RANGE = (-33.28, -33.28, -3, 33.28, 33.28, 1)
MAX_POINTS_PER_VOXEL = 128
FEATURE_CHANNELS = 16
FILTERS_PER_BLOCK = 3
PYRAMID_LAYERS = 1

SEQUENCE_LENGTH = 5

NSFP_FILTER_SIZE = 64
NSFP_NUM_LAYERS = 4

DEVICE = 'cuda'  #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sequence_loader = ArgoverseSequenceLoader('/bigdata/argoverse_lidar/train/')
# dataset = SequenceDataset(sequence_loader, SEQUENCE_LENGTH)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

joint_flow = JointFlow(VOXEL_SIZE, PSEUDO_IMAGE_DIMS, POINT_CLOUD_RANGE,
                       MAX_POINTS_PER_VOXEL, FEATURE_CHANNELS,
                       FILTERS_PER_BLOCK, PYRAMID_LAYERS, SEQUENCE_LENGTH,
                       NSFP_FILTER_SIZE, NSFP_NUM_LAYERS).to(DEVICE)

optimizer = torch.optim.Adam(joint_flow.parameters(), lr=1e-4)

global_step = 0
for sequence_idx, sequence_id in enumerate(sequence_loader.get_sequence_ids()):
    sequence = sequence_loader.load_sequence(sequence_id)

    num_steps = len(sequence) // SEQUENCE_LENGTH
    total_loss = 0
    for step in tqdm(range(num_steps), desc=f"Sequence {sequence_idx}"):
        offset = step * SEQUENCE_LENGTH
        subsequence = [sequence[offset + i] for i in range(SEQUENCE_LENGTH)]
        optimizer.zero_grad()
        subsequence_with_flow = joint_flow(subsequence)
        loss = 0
        loss += joint_flow.loss(subsequence_with_flow, 1)
        loss += joint_flow.loss(subsequence_with_flow, 2)
        writer.add_scalar('training_loss',
                          loss.item(),
                          global_step=global_step)
        global_step += 1
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Sequence {sequence_idx} loss: {total_loss / num_steps}")
