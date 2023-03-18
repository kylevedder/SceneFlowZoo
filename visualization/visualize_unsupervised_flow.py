import torch
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseUnsupervisedFlowSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm

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

    return PointCloud.from_array(pc0_points_lst)


sequence_loader = ArgoverseUnsupervisedFlowSequenceLoader(
    '/efs/argoverse2/val/', '/efs/argoverse2/val_nsfp_flow/')

sequence_id = sequence_loader.get_sequence_ids()[1]
print("Sequence ID: ", sequence_id)
sequence = sequence_loader.load_sequence(sequence_id)

# make open3d visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().point_size = 1.5
vis.get_render_option().background_color = (0, 0, 0)
vis.get_render_option().show_coordinate_frame = True
# set up vector
vis.get_view_control().set_up([0, 0, 1])

sequence_length = len(sequence)


def sequence_idx_to_color(idx):
    return [1 - idx / sequence_length, idx / sequence_length, 0]


frame_list = sequence.load_frame_list(None)
for idx, frame_dict in enumerate(tqdm.tqdm(frame_list[1:13])):
    pc = frame_dict['relative_pc']
    pc = voxel_restrict_pointcloud(pc)
    pose = frame_dict['relative_pose']
    flow = frame_dict['flow'][0]
    assert flow is not None, 'flow must not be None'
    flowed_pc = pc.flow(flow)

    # Add base point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.points)
    vis.add_geometry(pcd)

    # Add flowed point cloud
    if flowed_pc is not None:
        # flowed_pcd = o3d.geometry.PointCloud()
        # flowed_pcd.points = o3d.utility.Vector3dVector(flowed_pc.points)
        # flowed_pc_color = np.zeros_like(flowed_pc.points)
        # # Flow point cloud is blue
        # flowed_pc_color[:, 2] = 1.0
        # flowed_pcd.colors = o3d.utility.Vector3dVector(flowed_pc_color)
        # vis.add_geometry(flowed_pcd)

        # Add line set between pc0 and regressed pc1
        line_set = o3d.geometry.LineSet()
        assert len(pc) == len(
            flowed_pc
        ), f'pc and flowed_pc must have same length, but got {len(pc)} and {len(flowed_pc)}'
        line_set_points = np.concatenate([pc, flowed_pc], axis=0)

        lines = np.array([[i, i + len(pc)] for i in range(len(pc))])
        line_set.points = o3d.utility.Vector3dVector(line_set_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        # Line set is blue
        line_set.colors = o3d.utility.Vector3dVector(
            [[1, 0, 0] for _ in range(len(lines))])
        vis.add_geometry(line_set)

        # # Add gt pc from next frame
        # next_pc = frame_list[idx + 1]['relative_pc']
        # next_pcd = o3d.geometry.PointCloud()
        # next_pcd.points = o3d.utility.Vector3dVector(next_pc.points)
        # next_pc_color = np.zeros_like(next_pc.points)

        # next_pcd.colors = o3d.utility.Vector3dVector(next_pc_color)
        # vis.add_geometry(next_pcd)

    # Add center of mass
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    sphere.translate(pose.translation)
    sphere.paint_uniform_color(sequence_idx_to_color(idx))
    vis.add_geometry(sphere)

vis.run()
