import torch
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np

sequence_loader = ArgoverseSequenceLoader('/data/argoverse_lidar/')
sequence = sequence_loader.load_sequence(sequence_loader.get_sequence_ids()[0])

# make open3d visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().point_size = 1
vis.get_render_option().background_color = (0, 0, 0)
vis.get_render_option().show_coordinate_frame = True
# set up vector
vis.get_view_control().set_up([0, 0, 1])

for pc, pose in sequence:
    print(pc, pose)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.transform(pose).points)
    vis.add_geometry(pcd)

# Add sphere at 0,0,0
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
sphere.paint_uniform_color([1, 0, 0])
vis.add_geometry(sphere)


vis.run()
