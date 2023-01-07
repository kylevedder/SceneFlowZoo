import torch
from tqdm import tqdm
from dataloaders import ArgoverseSequenceLoader, ArgoverseSequence, ToTorchDataset

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

joint_flow = JointFlow(VOXEL_SIZE, PSEUDO_IMAGE_DIMS, POINT_CLOUD_RANGE,
                       MAX_POINTS_PER_VOXEL, FEATURE_CHANNELS,
                       FILTERS_PER_BLOCK, PYRAMID_LAYERS, NSFP_FILTER_SIZE,
                       NSFP_NUM_LAYERS).to(DEVICE)


def pc_to_torch(pc: PointCloud):
    return torch.from_numpy(pc.points).float().to(DEVICE)


def process_pc_pose(pc, pose):
    # The pc is in a global frame relative to the initial pose of the sequence.
    # The pose is the pose of the current point cloud relative to the initial pose of the sequence.
    # In order to voxelize the point cloud, we need to translate it to the origin.
    translated_pc = pc.translate(-pose.translation)

    # Pointcloud must be converted to torch tensor before passing to voxelizer.
    # The voxelizer expects a tensor of shape (N, 3) where N is the number of points.
    translated_pc_torch = pc_to_torch(translated_pc)

    translation_x_torch = torch.ones(
        (1, 1, PSEUDO_IMAGE_DIMS[0],
         PSEUDO_IMAGE_DIMS[1])).to(DEVICE) * pose.translation[0]
    translation_y_torch = torch.ones(
        (1, 1, PSEUDO_IMAGE_DIMS[0],
         PSEUDO_IMAGE_DIMS[1])).to(DEVICE) * pose.translation[1]

    pseudoimage = embedder(translated_pc_torch)
    pseudoimage = torch.cat(
        (pseudoimage, translation_x_torch, translation_y_torch), dim=1)
    latent = pyramid(pseudoimage)
    return latent


def warped_pc_loss(warped_pc_t,
                   pc_t1,
                   nsfp_param_list: list,
                   dist_threshold=2.0,
                   param_regularizer=10e-4):
    loss = 0
    batched_warped_pc_t = warped_pc_t.unsqueeze(0)
    batched_pc_t1 = pc_t1.unsqueeze(0)

    # Compute min distance between warped point cloud and point cloud at t+1.

    warped_pc_t_shape_tensor = torch.LongTensor([warped_pc_t.shape[0]]).to(
        batched_warped_pc_t.device)
    pc_t1_shape_tensor = torch.LongTensor([pc_t1.shape[0]
                                           ]).to(batched_pc_t1.device)
    warped_to_t1_knn = knn_points(p1=batched_warped_pc_t,
                                  p2=batched_pc_t1,
                                  lengths1=warped_pc_t_shape_tensor,
                                  lengths2=pc_t1_shape_tensor,
                                  K=1)
    warped_to_t1_distances = warped_to_t1_knn.dists[0]
    t1_to_warped_knn = knn_points(p1=batched_pc_t1,
                                  p2=batched_warped_pc_t,
                                  lengths1=pc_t1_shape_tensor,
                                  lengths2=warped_pc_t_shape_tensor,
                                  K=1)
    t1_to_warped_distances = t1_to_warped_knn.dists[0]
    # breakpoint()

    # Throw out distances that are too large (beyond the dist threshold).
    # warped_to_t1_distances[warped_to_t1_distances > dist_threshold] = 0
    # t1_to_warped_distances[t1_to_warped_distances > dist_threshold] = 0

    # Add up the distances.
    loss += torch.sum(warped_to_t1_distances) + torch.sum(
        t1_to_warped_distances)

    # L2 regularization on the neural scene flow prior parameters.
    for nsfp_params in nsfp_param_list:
        loss += torch.sum(nsfp_params**2 * param_regularizer)

    return loss


def loss_over_delta_time(delta: int):
    assert delta > 0, f"delta must be positive, got {delta}"
    assert delta < SEQUENCE_LENGTH, f"delta must be less than SEQUENCE_LENGTH, got {delta}"
    loss = 0
    for i in range(SEQUENCE_LENGTH - delta):
        pc_t, _ = sequence[offset + i]
        pc_t1, _ = sequence[offset + i + delta]
        pc_t_torch = pc_to_torch(pc_t)
        pc_t1_torch = pc_to_torch(pc_t1)

        warped_pc_t = pc_t_torch
        param_list = []
        for j in range(delta):
            nsfp_params = attention_output[0, (i + j) *
                                           nsfp.param_count:(i + j + 1) *
                                           nsfp.param_count]
            warped_pc_t = nsfp(warped_pc_t, nsfp_params)
            param_list.append(nsfp_params)
        loss += warped_pc_loss(warped_pc_t,
                               pc_t1_torch,
                               nsfp_param_list=param_list)
    return loss


optimizer = torch.optim.Adam(
    list(embedder.parameters()) + list(pyramid.parameters()) +
    list(nsfp.parameters()) + list(attention.parameters()),
    lr=1e-4)

global_step = 0
for sequence_idx, sequence_id in enumerate(sequence_loader.get_sequence_ids()):
    sequence = sequence_loader.load_sequence(sequence_id)

    num_steps = len(sequence) // SEQUENCE_LENGTH
    total_loss = 0
    for step in tqdm(range(num_steps), desc=f"Sequence {sequence_idx}"):
        offset = step * SEQUENCE_LENGTH
        optimizer.zero_grad()
        latent_list = []
        for i in range(SEQUENCE_LENGTH):
            pc, pose = sequence[offset + i]
            latent = process_pc_pose(pc, pose)
            latent_list.append(latent)
        latent_concat = torch.cat(latent_list, dim=1)

        attention_output = attention(latent_concat)
        assert attention_output.shape == (
            1, SEQUENCE_LENGTH * nsfp.param_count
        ), f"Attention output shape is {attention_output.shape}"

        loss = 0
        loss += loss_over_delta_time(1)
        loss += loss_over_delta_time(2)
        writer.add_scalar('training_loss',
                          loss.item(),
                          global_step=global_step)
        global_step += 1
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Sequence {sequence_idx} loss: {total_loss / num_steps}")
