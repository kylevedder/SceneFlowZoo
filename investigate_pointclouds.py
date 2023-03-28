import argparse
from pathlib import Path
import numpy as np
import tqdm
import joblib
import multiprocessing
from loader_utils import save_pickle
import open3d as o3d
from pointclouds import PointCloud
import torch

from dataloaders import ArgoverseRawSequenceLoader, WaymoSupervisedFlowSequenceLoader

# cli argument to pick argo or waymo
parser = argparse.ArgumentParser()
parser.add_argument('dataset',
                    type=str,
                    help='argo or waymo',
                    choices=['argo', 'waymo'])
parser.add_argument('--cpus',
                    type=int,
                    help='number of cpus to use',
                    default=multiprocessing.cpu_count() - 1)
args = parser.parse_args()

argoverse_data = Path("/efs/argoverse2")
waymo_data = Path("/efs/waymo_open_processed_flow")

from models.embedders import DynamicVoxelizer

from configs.pseudoimage import POINT_CLOUD_RANGE, VOXEL_SIZE

voxelizer = DynamicVoxelizer(voxel_size=VOXEL_SIZE,
                             point_cloud_range=POINT_CLOUD_RANGE)


def voxel_restrict_pointcloud(pc: PointCloud):
    pc0s = pc.points
    assert pc0s.ndim == 2, "pc0s must be a batch of 3D points"
    assert pc0s.shape[1] == 3, "pc0s must be a batch of 3D points"
    # Convert to torch tensor
    pc0s = pc0s.reshape(1, -1, 3)
    pc0s = torch.from_numpy(pc0s).float()

    pc0_voxel_infos_lst = voxelizer(pc0s)

    pc0_points_lst = [e["points"] for e in pc0_voxel_infos_lst]
    pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_voxel_infos_lst]

    # Convert to numpy
    pc0_points_lst = [e.numpy() for e in pc0_points_lst][0]
    pc0_valid_point_idxes = [e.numpy() for e in pc0_valid_point_idxes][0]

    return PointCloud.from_array(pc0_points_lst), pc0_valid_point_idxes


def get_pc_sizes(seq):
    pc_sizes = []
    for idx in range(len(seq)):
        frame = seq.load(idx, idx)
        pc = frame['relative_pc']
        pc, _ = voxel_restrict_pointcloud(pc)
        num_points = len(pc)
        if num_points < 100:
            print(seq.log_id, frame['log_idx'], num_points)
        pc_sizes.append(num_points)
    return pc_sizes


if args.dataset == 'waymo':
    seq_loader = WaymoSupervisedFlowSequenceLoader(waymo_data / "validation")
else:
    seq_loader = ArgoverseRawSequenceLoader(argoverse_data / "val")
seq_ids = seq_loader.get_sequence_ids()

if args.cpus > 1:
    # Use joblib to parallelize the loading of the sequences
    pc_size_lst = joblib.Parallel(n_jobs=args.cpus)(
        joblib.delayed(get_pc_sizes)(seq_loader.load_sequence(seq_id))
        for seq_id in tqdm.tqdm(seq_ids))
    pc_size_lst = [item for sublist in pc_size_lst for item in sublist]
else:
    pc_size_lst = []
    for seq_id in tqdm.tqdm(seq_ids):
        seq = seq_loader.load_sequence(seq_id)
        pc_sizes = get_pc_sizes(seq)
        pc_size_lst.extend(pc_sizes)

save_pickle(
    f"validation_results/{args.dataset}_validation_pointcloud_point_count.pkl",
    pc_size_lst)

print(np.mean(pc_size_lst), np.std(pc_size_lst), np.median(pc_size_lst))
