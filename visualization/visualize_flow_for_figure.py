import torch
import pandas as pd
import open3d as o3d
from bucketed_scene_flow_eval.datasets.argoverse2 import ArgoverseUnsupervisedFlowSequenceLoader, ArgoverseSupervisedSceneFlowSequenceLoader, ArgoverseSupervisedSceneFlowSequence
from bucketed_scene_flow_eval.datasets.waymoopen import WaymoSupervisedSceneFlowSequenceLoader, WaymoSupervisedSceneFlowSequence
import numpy as np
import tqdm
from pathlib import Path
from loader_utils import load_pickle, save_pickle
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default='argo',
                    help='waymo or argo',
                    choices=['waymo', 'argo'])
parser.add_argument('--sequence', type=str, default=None, help='waymo or argo')
args = parser.parse_args()

# sequence_loader = ArgoverseSequenceLoader('/bigdata/argoverse_lidar/train/')
# sequence_loader = ArgoverseSupervisedFlowSequenceLoader(
#     '/efs/argoverse2/val/', '/efs/argoverse2/val_sceneflow/')

# supervised_sequence_loader = ArgoverseSupervisedFlowSequenceLoader(
#     '/efs/argoverse2/val/', '/efs/argoverse2/val_sceneflow/')
# unsupervised_sequence_loader = ArgoverseUnsupervisedFlowSequenceLoader(
#     '/efs/argoverse2/val/', '/efs/argoverse2/val_distilation_flow/')

if args.dataset == "waymo":
    gt_sequence_loader = WaymoSupervisedSceneFlowSequenceLoader(
        '/efs/waymo_open_processed_flow/validation/')
    dist_sequence_loader = WaymoUnsupervisedSceneFlowSequenceLoader(
        '/efs/waymo_open_processed_flow/validation/',
        '/bigdata/waymo_open_processed_flow/val_distillation_out/')
    sup_sequence_loader = WaymoUnsupervisedSceneFlowSequenceLoader(
        '/efs/waymo_open_processed_flow/validation/',
        '/bigdata/waymo_open_processed_flow/val_supervised_out/')
elif args.dataset == "argo":
    gt_sequence_loader = ArgoverseSupervisedSceneFlowSequenceLoader(
        '/efs/argoverse2/val/', '/efs/argoverse2/val_sceneflow/')
    dist_sequence_loader = ArgoverseUnsupervisedFlowSequenceLoader(
        '/efs/argoverse2/val/', '/efs/argoverse2/val_distilation_out/')
    sup_sequence_loader = ArgoverseUnsupervisedFlowSequenceLoader(
        '/efs/argoverse2/val/', '/efs/argoverse2/val_supervised_out/')
else:
    raise ValueError(f'Unknown dataset {args.dataset}')

screenshot_path = Path() / 'screenshots'
screenshot_path.mkdir(exist_ok=True)

sequence_idx = 0

if args.sequence is not None:
    sequence_idx = gt_sequence_loader.get_sequence_ids().index(
        args.sequence)

draw_flow_lines_color = None

flow_type = 'gt'

starter_idx = 122


def load_sequence(sequence_idx: int):
    sequence_id = gt_sequence_loader.get_sequence_ids()[sequence_idx]
    assert sequence_id in set(
        dist_sequence_loader.get_sequence_ids()
    ), f"sequence_id {sequence_id} not in unsupervised_sequence_loader.get_sequence_ids() {dist_sequence_loader.get_sequence_ids()}"
    print("Sequence ID: ", sequence_id, "Sequence idx: ", sequence_idx)
    supervised_sequence = gt_sequence_loader.load_sequence(sequence_id)
    our_sequence = dist_sequence_loader.load_sequence(sequence_id)
    sup_sequence = sup_sequence_loader.load_sequence(sequence_id)

    before_human_load = time.time()
    supervised_frame_list = supervised_sequence.load_frame_list(0)
    before_our_load = time.time()
    print(f"Time to load human: {before_our_load - before_human_load}")
    our_frame_list = our_sequence.load_frame_list(0)
    before_sup_load = time.time()
    print(f"Time to load our: {before_sup_load - before_our_load}")
    sup_frame_list = sup_sequence.load_frame_list(0)
    after_sup_load = time.time()
    print(f"Time to load sup: {after_sup_load - before_sup_load}")

    return supervised_frame_list, our_frame_list, sup_frame_list


# make open3d visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.get_render_option().point_size = 2
vis.get_render_option().background_color = (1, 1, 1)
vis.get_render_option().show_coordinate_frame = False
# Set camera parameters

supervised_frame_list, our_frame_list, sup_frame_list = load_sequence(
    sequence_idx)


def toggle_flow_lines(vis):
    global draw_flow_lines_color

    if draw_flow_lines_color is None:
        draw_flow_lines_color = 'red'
    elif draw_flow_lines_color == 'red':
        draw_flow_lines_color = 'blue'
    elif draw_flow_lines_color == 'blue':
        draw_flow_lines_color = 'green'
    elif draw_flow_lines_color == 'green':
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


def toggle_flow_type(vis):
    global flow_type
    if flow_type == 'gt':
        flow_type = 'ours'
    elif flow_type == 'ours':
        flow_type = 'sup'
    elif flow_type == 'sup':
        flow_type = 'gt'
    else:
        raise ValueError(f'Invalid flow_type: {flow_type}')
    vis.clear_geometries()
    draw_frames(reset_view=False)


def increase_starter_idx(vis):
    global starter_idx
    starter_idx += 1
    if starter_idx >= min(len(supervised_frame_list), len(our_frame_list)) - 1:
        starter_idx = 0
    print(starter_idx)
    vis.clear_geometries()
    draw_frames(reset_view=False)


def decrease_starter_idx(vis):
    global starter_idx
    starter_idx -= 1
    if starter_idx < 0:
        starter_idx = min(len(supervised_frame_list), len(our_frame_list)) - 2
    print(starter_idx)
    vis.clear_geometries()
    draw_frames(reset_view=False)


def starter_idx_beginning(vis):
    global starter_idx
    starter_idx = 0
    print(starter_idx)
    vis.clear_geometries()
    draw_frames(reset_view=False)


def starter_idx_end(vis):
    global starter_idx
    starter_idx = min(len(supervised_frame_list), len(our_frame_list)) - 2
    print(starter_idx)
    vis.clear_geometries()
    draw_frames(reset_view=False)


def save_screenshot(vis):
    save_name = screenshot_path / f'{gt_sequence_loader.get_sequence_ids()[sequence_idx]}_{starter_idx}_{time.time():03f}.png'
    vis.capture_screen_image(str(save_name))


def decrease_sequence_idx(vis):
    global sequence_idx
    global supervised_frame_list
    global our_frame_list
    global sup_frame_list
    sequence_idx -= 1
    supervised_frame_list, our_frame_list, sup_frame_list = load_sequence(
        sequence_idx)
    vis.clear_geometries()
    draw_frames(reset_view=True)


def increase_sequence_idx(vis):
    global sequence_idx
    global supervised_frame_list
    global our_frame_list
    global sup_frame_list
    sequence_idx += 1
    supervised_frame_list, our_frame_list, sup_frame_list = load_sequence(
        sequence_idx)
    vis.clear_geometries()
    draw_frames(reset_view=True)


vis.register_key_callback(ord("F"), toggle_flow_lines)
vis.register_key_callback(ord("G"), toggle_flow_type)
# left arrow decrease starter_idx
vis.register_key_callback(263, decrease_starter_idx)
# # right arrow increase starter_idx
vis.register_key_callback(262, increase_starter_idx)
# Up arrow for next sequence
vis.register_key_callback(265, increase_sequence_idx)
# Down arrow for prior sequence
vis.register_key_callback(264, decrease_sequence_idx)
# Home key for beginning of sequence
vis.register_key_callback(268, starter_idx_beginning)
# End key for end of sequence
vis.register_key_callback(269, starter_idx_end)
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
    limited_our_frame_list = limit_frames(our_frame_list)
    limited_sup_frame_list = limit_frames(sup_frame_list)
    #color_scalar = np.linspace(0.5, 1, len(limited_supervised_frame_list))
    # Green then blue
    color_scalar = [[0, 1, 0], [0, 0, 1]]
    frame_dict_list = list(
        zip(limited_supervised_frame_list, limited_our_frame_list,
            limited_sup_frame_list))
    for idx, (supervised_frame_dict, our_frame_dict,
              sup_frame_dict) in enumerate(tqdm.tqdm(frame_dict_list)):
        supervised_pc = supervised_frame_dict['relative_pc']
        pose = supervised_frame_dict['relative_pose']
        supervised_flowed_pc = supervised_frame_dict['relative_flowed_pc']
        classes = supervised_frame_dict['pc_classes']

        our_pc_valid_idxes = our_frame_dict['valid_idxes']
        our_pc = our_frame_dict['relative_pc'][our_pc_valid_idxes]

        sup_pc_valid_idxes = sup_frame_dict['valid_idxes']
        sup_pc = sup_frame_dict['relative_pc'][sup_pc_valid_idxes]

        # Add base point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(supervised_pc.points)
        pc_color = np.ones_like(supervised_pc.points) * color_scalar[idx]
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
        vis.add_geometry(pcd, reset_bounding_box=reset_view)

        our_flow = our_frame_dict['flow']
        if our_flow is None:
            continue
        if our_flow.ndim == 3:
            our_flow = our_flow.squeeze(0)
        our_flowed_pc = our_pc + our_flow

        sup_flow = sup_frame_dict['flow']
        if sup_flow is None:
            continue
        if sup_flow.ndim == 3:
            sup_flow = sup_flow.squeeze(0)
        sup_flowed_pc = sup_pc + sup_flow

        # Add flowed point cloud
        if supervised_flowed_pc is not None and idx < len(frame_dict_list) - 1:
            print("sequence_idx", sequence_idx, "starter_idx:", starter_idx,
                  "Flow type: ", flow_type)

            if draw_flow_lines_color is not None:

                if flow_type == 'gt':
                    line_set = make_lineset(supervised_pc.points,
                                            supervised_flowed_pc.points,
                                            draw_flow_lines_color)
                elif flow_type == 'ours':
                    line_set = make_lineset(our_pc, our_flowed_pc,
                                            draw_flow_lines_color)
                elif flow_type == 'sup':
                    line_set = make_lineset(sup_pc, sup_flowed_pc,
                                            draw_flow_lines_color)
                else:
                    raise ValueError(f'Unknown flow_type {flow_type}')
                vis.add_geometry(line_set, reset_bounding_box=reset_view)


draw_frames(reset_view=True)

vis.run()
