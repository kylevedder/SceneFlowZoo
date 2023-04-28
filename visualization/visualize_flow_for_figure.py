import torch
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseRawSequenceLoader, ArgoverseSupervisedFlowSequenceLoader, WaymoSupervisedFlowSequence, WaymoSupervisedFlowSequenceLoader, WaymoUnsupervisedFlowSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm
from pathlib import Path
from loader_utils import load_pickle, save_pickle
import time

# sequence_loader = ArgoverseSequenceLoader('/bigdata/argoverse_lidar/train/')
# sequence_loader = ArgoverseSupervisedFlowSequenceLoader(
#     '/efs/argoverse2/val/', '/efs/argoverse2/val_sceneflow/')
supervised_sequence_loader = WaymoSupervisedFlowSequenceLoader(
    '/efs/waymo_open_processed_flow/training/')
unsupervised_sequence_loader = WaymoUnsupervisedFlowSequenceLoader(
    '/efs/waymo_open_processed_flow/training/',
    '/efs/waymo_open_processed_flow/train_nsfp_flow/')

camera_params_path = Path() / 'camera_params.pkl'
screenshot_path = Path() / 'screenshots'
screenshot_path.mkdir(exist_ok=True)

sequence_id = supervised_sequence_loader.get_sequence_ids()[1]
assert sequence_id in set(
    unsupervised_sequence_loader.get_sequence_ids()
), f"sequence_id {sequence_id} not in unsupervised_sequence_loader.get_sequence_ids() {unsupervised_sequence_loader.get_sequence_ids()}"
print("Sequence ID: ", sequence_id)
supervised_sequence = supervised_sequence_loader.load_sequence(sequence_id)
unsupervised_sequence = unsupervised_sequence_loader.load_sequence(sequence_id)

# make open3d visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.get_render_option().point_size = 0.5
vis.get_render_option().background_color = (0, 0, 0)
vis.get_render_option().show_coordinate_frame = False
# Set camera parameters


def load_camera_params(view_control: o3d.visualization.ViewControl):
    if not camera_params_path.exists():
        return None
    param_dict = load_pickle(camera_params_path)
    intrinsics = param_dict['intrinsics']
    extrinsics = param_dict['extrinsics']
    pinhole_camera_params = view_control.convert_to_pinhole_camera_parameters()
    pinhole_camera_params.intrinsic.intrinsic_matrix = intrinsics
    pinhole_camera_params.extrinsic = extrinsics
    return pinhole_camera_params


def save_camera_params(camera_params: o3d.camera.PinholeCameraParameters):
    param_dict = {}
    param_dict['intrinsics'] = camera_params.intrinsic.intrinsic_matrix
    param_dict['extrinsics'] = camera_params.extrinsic
    save_pickle(camera_params_path, param_dict)


ctr = vis.get_view_control()
camera_params = load_camera_params(ctr)
if camera_params is not None:
    ctr.convert_from_pinhole_camera_parameters(camera_params,
                                               allow_arbitrary=True)
else:
    print("No camera params found, using default view")
# ctr.look_at([0, 0, 0])

sequence_length = len(supervised_sequence)


def sequence_idx_to_color(idx):
    return [1 - idx / sequence_length, idx / sequence_length, 0]


supervised_frame_list = supervised_sequence.load_frame_list(0)
unsupervised_frame_list = unsupervised_sequence.load_frame_list(0)

draw_flow_lines_color = None

use_gt_flow = True

starter_idx = 164


def toggle_flow_lines(vis):
    global draw_flow_lines_color

    if draw_flow_lines_color is None:
        draw_flow_lines_color = 'green'
    elif draw_flow_lines_color == 'green':
        draw_flow_lines_color = 'blue'
    elif draw_flow_lines_color == 'blue':
        draw_flow_lines_color = 'red'
    elif draw_flow_lines_color == 'red':
        draw_flow_lines_color = 'purple'
    elif draw_flow_lines_color == 'purple':
        draw_flow_lines_color = 'yellow'
    elif draw_flow_lines_color == 'yellow':
        draw_flow_lines_color = 'orange'
    elif draw_flow_lines_color == 'orange':
        draw_flow_lines_color = None
    else:
        raise ValueError(
            f'Invalid draw_flow_lines_color: {draw_flow_lines_color}')
    vis.clear_geometries()
    draw_frames(reset_view=False)


def toggle_gt_flow(vis):
    global use_gt_flow
    use_gt_flow = not use_gt_flow
    vis.clear_geometries()
    draw_frames(reset_view=False)


def increase_starter_idx(vis):
    global starter_idx
    starter_idx += 1
    print(starter_idx)
    vis.clear_geometries()
    draw_frames(reset_view=False)


def decrease_starter_idx(vis):
    global starter_idx
    starter_idx -= 1
    print(starter_idx)
    vis.clear_geometries()
    draw_frames(reset_view=False)


def save_camera_params_callback(vis):
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    save_camera_params(params)


def save_screenshot(vis):
    save_name = screenshot_path / f'{sequence_id}_{starter_idx}_{time.time():03f}.png'
    vis.capture_screen_image(str(save_name))


vis.register_key_callback(ord("F"), toggle_flow_lines)
vis.register_key_callback(ord("G"), toggle_gt_flow)
# left arrow decrease starter_idx
vis.register_key_callback(263, decrease_starter_idx)
# # right arrow increase starter_idx
vis.register_key_callback(262, increase_starter_idx)
vis.register_key_callback(ord("C"), save_camera_params_callback)
vis.register_key_callback(ord("S"), save_screenshot)


def make_lineset(pc, flowed_pc, draw_color):
    line_set = o3d.geometry.LineSet()
    assert len(pc) == len(
        flowed_pc
    ), f'pc and flowed_pc must have same length, but got {len(pc)} and {len(flowed_pc)}'
    line_set_points = np.concatenate([pc, flowed_pc], axis=0)

    lines = np.array([[i, i + len(pc)] for i in range(len(pc))])
    line_set.points = o3d.utility.Vector3dVector(line_set_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    draw_color = {
        'green': [0, 1, 0],
        'blue': [0, 0, 1],
        'red': [1, 0, 0],
        'yellow': [1, 1, 0],
        'orange': [1, 0.5, 0],
        'purple': [0.5, 0, 1],
        'pink': [1, 0, 1],
        'cyan': [0, 1, 1],
        'white': [1, 1, 1],
    }[draw_flow_lines_color]
    line_set.colors = o3d.utility.Vector3dVector(
        [draw_color for _ in range(len(lines))])
    return line_set


def limit_frames(frame_lst):
    return frame_lst[starter_idx:starter_idx + 2]


def draw_frames(reset_view=False):
    limited_supervised_frame_list = limit_frames(supervised_frame_list)
    limited_unsupervised_frame_list = limit_frames(unsupervised_frame_list)
    #color_scalar = np.linspace(0.5, 1, len(limited_supervised_frame_list))
    # Green then blue
    color_scalar = [[0, 1, 0], [0, 0, 1]]
    frame_dict_list = list(
        zip(limited_supervised_frame_list, limited_unsupervised_frame_list))
    for idx, (supervised_frame_dict, unsupervised_frame_dict) in enumerate(
            tqdm.tqdm(frame_dict_list)):
        supervised_pc = supervised_frame_dict['relative_pc']
        pose = supervised_frame_dict['relative_pose']
        supervised_flowed_pc = supervised_frame_dict['relative_flowed_pc']
        classes = supervised_frame_dict['pc_classes']

        unsupervised_pc_valid_idxes = unsupervised_frame_dict['valid_idxes']
        unsupervised_pc = unsupervised_frame_dict['relative_pc'][
            unsupervised_pc_valid_idxes]

        unsupervised_flow = unsupervised_frame_dict['flow'].squeeze(0)
        unsupervised_flowed_pc = unsupervised_pc + unsupervised_flow

        # Add base point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(supervised_pc.points)
        pc_color = np.ones_like(supervised_pc.points) * color_scalar[idx]
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
        vis.add_geometry(pcd, reset_bounding_box=reset_view)

        # Add flowed point cloud
        if supervised_flowed_pc is not None and idx < len(
                supervised_frame_list) - 1:
            print(
                "Max flow magnitude:",
                supervised_pc.matched_point_distance(
                    supervised_flowed_pc).max())

            if draw_flow_lines_color is not None:

                if use_gt_flow:
                    line_set = make_lineset(supervised_pc.points,
                                            supervised_flowed_pc.points,
                                            draw_flow_lines_color)
                else:
                    line_set = make_lineset(unsupervised_pc,
                                            unsupervised_flowed_pc,
                                            draw_flow_lines_color)
                vis.add_geometry(line_set, reset_bounding_box=reset_view)


draw_frames(reset_view=True)

vis.run()
