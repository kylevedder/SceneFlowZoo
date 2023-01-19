import torch
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseSequenceLoader, ArgoverseFlowSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm

# sequence_loader = ArgoverseSequenceLoader('/bigdata/argoverse_lidar/train/')
sequence_loader = ArgoverseFlowSequenceLoader(
    '/efs/argoverse2/val/', '/efs/argoverse2/val_sceneflow/')
sequence = sequence_loader.load_sequence(
    sequence_loader.get_sequence_ids()[29])

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


frame_list = sequence.load_frame_list(0)
for idx, (pc, pose, flowed_pc, classes,
          is_grounds) in enumerate(tqdm.tqdm(frame_list[:1])):

    # Add base point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.points)
    pc_color = np.zeros_like(pc.points)
    pc_color[:, 0] = 1.0
    pcd.colors = o3d.utility.Vector3dVector(pc_color)
    vis.add_geometry(pcd)

    # Add flowed point cloud
    if flowed_pc is not None:
        flowed_pcd = o3d.geometry.PointCloud()
        flowed_pcd.points = o3d.utility.Vector3dVector(flowed_pc.points)
        flowed_pc_color = np.zeros_like(flowed_pc.points)
        flowed_pc_color[:, 2] = 1.0
        flowed_pcd.colors = o3d.utility.Vector3dVector(flowed_pc_color)
        vis.add_geometry(flowed_pcd)

        # Add gt pc from next frame
        next_pc, _, _, _, _ = frame_list[idx + 1]
        next_pcd = o3d.geometry.PointCloud()
        next_pcd.points = o3d.utility.Vector3dVector(next_pc.points)
        next_pc_color = np.zeros_like(next_pc.points)
        next_pc_color[:, 1] = 1.0
        next_pcd.colors = o3d.utility.Vector3dVector(next_pc_color)
        vis.add_geometry(next_pcd)

    # Add center of mass
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    sphere.translate(pose.translation)
    sphere.paint_uniform_color(sequence_idx_to_color(idx))
    vis.add_geometry(sphere)

vis.run()
