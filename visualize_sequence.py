import torch
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm

sequence_loader = ArgoverseSequenceLoader('/bigdata/argoverse_lidar/train/')
sequence = sequence_loader.load_sequence(
    sequence_loader.get_sequence_ids()[29])

# make open3d visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().point_size = 1
vis.get_render_option().background_color = (0, 0, 0)
vis.get_render_option().show_coordinate_frame = True
# set up vector
vis.get_view_control().set_up([0, 0, 1])

sequence_length = len(sequence)


def sequence_idx_to_color(idx):
    return [1 - idx / sequence_length, idx / sequence_length, 0]


for idx, (pc, pose) in enumerate(tqdm.tqdm(sequence.load_frame_list(0))):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.points)
    vis.add_geometry(pcd)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    sphere.translate(pose.translation)
    sphere.paint_uniform_color(sequence_idx_to_color(idx))
    vis.add_geometry(sphere)

vis.run()
