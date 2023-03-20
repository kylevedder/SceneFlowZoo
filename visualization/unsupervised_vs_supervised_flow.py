import torch
import pandas as pd
from dataloaders import ArgoverseUnsupervisedFlowSequenceLoader, ArgoverseSupervisedFlowSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm
import argparse
from pathlib import Path
import joblib
import multiprocessing

from models.embedders import DynamicVoxelizer

from configs.pseudoimage import POINT_CLOUD_RANGE, VOXEL_SIZE

parser = argparse.ArgumentParser()
# dataset type (train, val), default to val
parser.add_argument('--dataset_type',
                    type=str,
                    default='val',
                    choices=['train', 'val'])
# select every nth frame, default to 1
parser.add_argument('--step_size',
                    type=int,
                    default=5,
                    help='select every nth frame')
# num workers
parser.add_argument('--num_workers',
                    type=int,
                    default=multiprocessing.cpu_count(),
                    help='number of workers')
args = parser.parse_args()

assert args.step_size > 0, 'step_size must be positive'

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


def get_pcs_flows_pose(unsupervised_frame_dict, supervised_frame_dict):

    # Load unsupervised data
    unrestricted_pc = unsupervised_frame_dict['relative_pc']

    # pc, valid_points_pc = voxel_restrict_pointcloud(unrestricted_pc)
    pose = unsupervised_frame_dict['relative_pose']
    unsupervised_flow = unsupervised_frame_dict['flow'][0]
    valid_idxes = unsupervised_frame_dict['valid_idxes']
    pc = unrestricted_pc.mask_points(valid_idxes)
    assert unsupervised_flow is not None, 'flow must not be None'
    assert pc.shape == unsupervised_flow.shape, f"pc and flow must have same shape, but got {pc.shape} and {unsupervised_flow.shape}"
    unsupervised_flowed_pc = pc.flow(unsupervised_flow)

    # Load supervised data
    full_supervised_flowed_pc = supervised_frame_dict['relative_flowed_pc']
    full_pc_classes = supervised_frame_dict['pc_classes']
    # Extract the full PC flow.
    supervised_flow = full_supervised_flowed_pc.points - unrestricted_pc.points
    supervised_flow = supervised_flow[valid_idxes]
    pc_classes = full_pc_classes[valid_idxes]
    supervised_flowed_pc = pc.flow(supervised_flow)

    assert unsupervised_flowed_pc.points.shape == supervised_flowed_pc.points.shape, f"unsupervised_flowed_pc and supervised_flowed_pc must have same shape, but got {unsupervised_flowed_pc.points.shape} and {supervised_flowed_pc.points.shape}"

    return unsupervised_flowed_pc, unsupervised_flow, supervised_flowed_pc, supervised_flow, pc_classes


def accumulate_flow_scatter(supervised_flow,
                            unsupervised_flow,
                            classes,
                            grid_radius_meters=2,
                            cells_per_meter=100):
    # import matplotlib.pyplot as plt
    not_background_mask = (classes >= 0)

    supervised_flow = supervised_flow[not_background_mask][:, :2]
    unsupervised_flow = unsupervised_flow[not_background_mask][:, :2]

    # Canonocalize the supervised flows so that ground truth is all pointing in the positive X direction
    supervised_norm = np.linalg.norm(supervised_flow, axis=1)
    non_zero_mask = (supervised_norm > 0.001)

    rotatable_supervised_flow = supervised_flow[non_zero_mask]
    rotatable_unsupervised_flow = unsupervised_flow[non_zero_mask]

    # Compute the angle between the supervised flow and the positive X axis
    supervised_angles = np.arctan2(rotatable_supervised_flow[:, 0],
                                   rotatable_supervised_flow[:, 1])

    # Compute the rotation matrix
    rotation_matrix = np.array(
        [[np.cos(supervised_angles), -np.sin(supervised_angles)],
         [np.sin(supervised_angles),
          np.cos(supervised_angles)]]).transpose(2, 0, 1)

    # Rotate the supervised flow
    supervised_flow[non_zero_mask] = np.einsum('ijk,ik->ij', rotation_matrix,
                                               rotatable_supervised_flow)

    # Rotate the unsupervised flow
    unsupervised_flow[non_zero_mask] = np.einsum('ijk,ik->ij', rotation_matrix,
                                                 rotatable_unsupervised_flow)

    # Compute the flow difference
    flow_error_xy = (unsupervised_flow - supervised_flow)

    # Create the grid
    accumulation_grid = np.zeros(
        (grid_radius_meters * 2 * cells_per_meter + 1,
         grid_radius_meters * 2 * cells_per_meter + 1))

    # Compute the grid indices for flow_error_xy using np digitize
    grid_indices_x = np.digitize(
        flow_error_xy[:, 0],
        bins=np.linspace(-grid_radius_meters, grid_radius_meters,
                         grid_radius_meters * 2 * cells_per_meter + 1)) - 1
    grid_indices_y = np.digitize(
        flow_error_xy[:, 1],
        bins=np.linspace(-grid_radius_meters, grid_radius_meters,
                         grid_radius_meters * 2 * cells_per_meter + 1)) - 1

    grid_indices = np.array([grid_indices_x, grid_indices_y]).T
    unique_grid_indices, unique_counts = np.unique(grid_indices,
                                                   axis=0,
                                                   return_counts=True)

    accumulation_grid[unique_grid_indices[:, 0],
                      unique_grid_indices[:, 1]] = unique_counts

    # plt.matshow(accumulation_grid)
    # plt.xticks(
    #     np.linspace(0, grid_radius_meters * 2 * cells_per_meter,
    #                 grid_radius_meters * 2 + 1),
    #     np.linspace(-grid_radius_meters, grid_radius_meters,
    #                 grid_radius_meters * 2 + 1))
    # plt.yticks(
    #     np.linspace(0, grid_radius_meters * 2 * cells_per_meter,
    #                 grid_radius_meters * 2 + 1),
    #     np.linspace(-grid_radius_meters, grid_radius_meters,
    #                 grid_radius_meters * 2 + 1))

    # plt.colorbar()
    # plt.show()

    return accumulation_grid


def process_sequence(sequence_id):

    unsupervised_sequence = unsupervised_sequence_loader.load_sequence(
        sequence_id)
    supervised_sequence = supervised_sequence_loader.load_sequence(sequence_id)

    unsupervised_frame_list = unsupervised_sequence.load_frame_list(None)
    supervised_frame_list = supervised_sequence.load_frame_list(None)

    zipped_frame_list = list(
        zip(unsupervised_frame_list,
            supervised_frame_list))[:-1][::args.step_size]

    accumulation_grid_lst = []
    for idx, (unsupervised_frame_dict,
              supervised_frame_dict) in enumerate(zipped_frame_list):

        step_idx = idx * args.step_size
        unsupervised_flowed_pc, unsupervised_flow, supervised_flowed_pc, supervised_flow, pc_classes = get_pcs_flows_pose(
            unsupervised_frame_dict, supervised_frame_dict)

        accumulation_grid_lst.append(
            accumulate_flow_scatter(supervised_flow, unsupervised_flow,
                                    pc_classes))

    accumulation_grid = np.sum(accumulation_grid_lst, axis=0)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / f'{sequence_id}_error_distribution.npy',
            accumulation_grid)


data_root = Path('/efs/argoverse2/')

unsupervised_sequence_loader = ArgoverseUnsupervisedFlowSequenceLoader(
    data_root / args.dataset_type,
    data_root / f'{args.dataset_type}_nsfp_flow/')

supervised_sequence_loader = ArgoverseSupervisedFlowSequenceLoader(
    data_root / args.dataset_type,
    data_root / f'{args.dataset_type}_sceneflow/')

save_dir = data_root / f'{args.dataset_type}_unsupervised_vs_supervised_flow'

unsupervised_sequences = set(unsupervised_sequence_loader.get_sequence_ids())
supervised_sequences = set(supervised_sequence_loader.get_sequence_ids())

overlapping_sequences = unsupervised_sequences.intersection(
    supervised_sequences)

print("Number of overlapping sequences: ", len(overlapping_sequences))

# Use joblib to parallelize the processing of each sequence
joblib.Parallel(n_jobs=args.num_workers, verbose=10)(
    joblib.delayed(process_sequence)(sequence_id)
    for sequence_id in tqdm.tqdm(sorted(overlapping_sequences)))

# for sequence_id in sorted(overlapping_sequences):
#     process_sequence(sequence_id)