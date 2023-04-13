import torch
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseRawSequenceLoader, ArgoverseSupervisedFlowSequenceLoader, WaymoSupervisedFlowSequence, WaymoSupervisedFlowSequenceLoader, WaymoUnsupervisedFlowSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm

# sequence_loader = ArgoverseSequenceLoader('/bigdata/argoverse_lidar/train/')
# sequence_loader = ArgoverseSupervisedFlowSequenceLoader(
#     '/efs/argoverse2/val/', '/efs/argoverse2/val_sceneflow/')
sequence_loader = WaymoSupervisedFlowSequenceLoader(
    '/efs/waymo_open_processed_flow/training/')

sequence_id = sequence_loader.get_sequence_ids()[1]
print("Sequence ID: ", sequence_id)
sequence = sequence_loader.load_sequence(sequence_id)

# make open3d visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.get_render_option().point_size = 1.5
vis.get_render_option().background_color = (0, 0, 0)
vis.get_render_option().show_coordinate_frame = True
# set up vector
vis.get_view_control().set_up([0, 0, 1])

sequence_length = len(sequence)


def sequence_idx_to_color(idx):
    return [1 - idx / sequence_length, idx / sequence_length, 0]


frame_list = sequence.load_frame_list(0)[125:127]


draw_flow_lines_color = None


def toggle_flow_lines(vis ):
    global draw_flow_lines_color

    if draw_flow_lines_color is None:
        draw_flow_lines_color = 'green'
    elif draw_flow_lines_color == 'green':
        draw_flow_lines_color = 'blue'
    elif draw_flow_lines_color == 'blue':
        draw_flow_lines_color = None
    else:
        raise ValueError(f'Invalid draw_flow_lines_color: {draw_flow_lines_color}')
    vis.clear_geometries()
    draw_frames(reset_view=False)


vis.register_key_callback(ord("F"), toggle_flow_lines)



def draw_frames(reset_view=False):
    color_scalar = np.linspace(0.5, 1, len(frame_list))
    for idx, frame_dict in enumerate(tqdm.tqdm(frame_list)):
        pc = frame_dict['relative_pc']
        pose = frame_dict['relative_pose']
        flowed_pc = frame_dict['relative_flowed_pc']
        classes = frame_dict['pc_classes']

        # Add base point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)
        pc_color = np.ones_like(pc.points) * color_scalar[idx]
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
        vis.add_geometry(pcd, reset_bounding_box=reset_view)

        # Add flowed point cloud
        if flowed_pc is not None and idx < len(frame_list) - 1:
            print("Max flow magnitude:",
                pc.matched_point_distance(flowed_pc).max())

            if draw_flow_lines_color is not None:
                line_set = o3d.geometry.LineSet()
                assert len(pc) == len(
                    flowed_pc
                ), f'pc and flowed_pc must have same length, but got {len(pc)} and {len(flowed_pc)}'
                line_set_points = np.concatenate([pc, flowed_pc], axis=0)

                lines = np.array([[i, i + len(pc)] for i in range(len(pc))])
                line_set.points = o3d.utility.Vector3dVector(line_set_points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                draw_color = [0, 1, 0] if draw_flow_lines_color == 'green' else [0, 0, 1]
                line_set.colors = o3d.utility.Vector3dVector(
                    [draw_color for _ in range(len(lines))])
                vis.add_geometry(line_set, reset_bounding_box=reset_view)

draw_frames(reset_view=True)

vis.run()
