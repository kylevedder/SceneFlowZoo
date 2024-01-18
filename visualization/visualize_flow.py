import torch
import pandas as pd
import open3d as o3d
from bucketed_scene_flow_eval.datasets.argoverse2 import ArgoverseSceneFlowSequenceLoader
import numpy as np
import tqdm
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sequence_id", type=str, default="0b86f508-5df9-4a46-bc59-5b9536dbde9f")
parser.add_argument("--sequence_start_idx", type=int, default=125)
# Path to the sequence folder
parser.add_argument("--sequence_folder", type=str, default="/efs/argoverse2/val/")
# Path to the flow folder
parser.add_argument(
    "--flow_folder",
    type=str,
    default="/efs/argoverse2/val_sceneflow_feather/",
    help="Path to the flow folder. This can be the ground truth flow, or dumped results from a model.",
)
args = parser.parse_args()

# sequence_loader = ArgoverseSequenceLoader('/bigdata/argoverse_lidar/train/')
sequence_loader = ArgoverseSceneFlowSequenceLoader(
    args.sequence_folder,
    args.flow_folder,
    use_gt_flow=False,
)

# sequence_id = sequence_loader.get_sequence_ids()[2]
sequence_id = args.sequence_id
print("Sequence ID: ", sequence_id)
sequence = sequence_loader.load_sequence(sequence_id)

# make open3d visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.get_render_option().point_size = 1.5
vis.get_render_option().background_color = (0, 0, 0)
# vis.get_render_option().show_coordinate_frame = True
# set up vector
vis.get_view_control().set_up([0, 0, 1])

sequence_length = len(sequence)


def sequence_idx_to_color(idx):
    return [1 - idx / sequence_length, idx / sequence_length, 0]


screenshot_path = Path() / "screenshots"
screenshot_path.mkdir(exist_ok=True)
draw_flow_lines_color = "green"
full_frame_list = sequence.load_frame_list(0)
starter_idx = args.sequence_start_idx


def increase_starter_idx(vis):
    global starter_idx
    starter_idx += 1
    if starter_idx >= len(full_frame_list) - 1:
        starter_idx = 0
    print("Index: ", starter_idx)
    vis.clear_geometries()
    draw_frames(reset_view=False)


def decrease_starter_idx(vis):
    global starter_idx
    starter_idx -= 1
    if starter_idx < 0:
        starter_idx = len(full_frame_list) - 2
    print("Index: ", starter_idx)
    vis.clear_geometries()
    draw_frames(reset_view=False)


def toggle_flow_lines(vis):
    global draw_flow_lines_color

    if draw_flow_lines_color is None:
        draw_flow_lines_color = "green"
    elif draw_flow_lines_color == "green":
        draw_flow_lines_color = "blue"
    elif draw_flow_lines_color == "blue":
        draw_flow_lines_color = None
    else:
        raise ValueError(f"Invalid draw_flow_lines_color: {draw_flow_lines_color}")
    vis.clear_geometries()
    draw_frames(reset_view=False)


def save_screenshot(vis):
    save_name = screenshot_path / sequence_id / f"{starter_idx:06d}.png"
    save_name.parent.mkdir(exist_ok=True, parents=True)
    vis.capture_screen_image(str(save_name))


vis.register_key_callback(ord("F"), toggle_flow_lines)
# left arrow decrease starter_idx
vis.register_key_callback(263, decrease_starter_idx)
# right arrow increase starter_idx
vis.register_key_callback(262, increase_starter_idx)
vis.register_key_callback(ord("S"), save_screenshot)

print("#############################################################")
print("Flow moves from the gray point cloud to the white point cloud\n")
print("Press F to toggle flow lines")
print("Press left or right arrow to change starter_idx")
print(f"Press S to save screenshot (saved to {screenshot_path.absolute()})")
print("#############################################################")


def draw_frames(reset_view=False):
    frame_list = full_frame_list[starter_idx : starter_idx + 2]
    color_scalar = np.linspace(0.5, 1, len(frame_list))
    for idx, frame_dict in enumerate(frame_list):
        pc = frame_dict["relative_pc"]
        pose = frame_dict["relative_pose"]
        flowed_pc = frame_dict["relative_flowed_pc"]
        # classes = frame_dict['pc_classes']

        # Add base point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)
        pc_color = np.ones_like(pc.points) * color_scalar[idx]
        pcd.colors = o3d.utility.Vector3dVector(pc_color)
        vis.add_geometry(pcd, reset_bounding_box=reset_view)

        # Add flowed point cloud
        if flowed_pc is not None and idx < len(frame_list) - 1:
            if draw_flow_lines_color is not None:
                line_set = o3d.geometry.LineSet()
                assert len(pc) == len(
                    flowed_pc
                ), f"pc and flowed_pc must have same length, but got {len(pc)} and {len(flowed_pc)}"
                line_set_points = np.concatenate([pc, flowed_pc], axis=0)

                lines = np.array([[i, i + len(pc)] for i in range(len(pc))])
                line_set.points = o3d.utility.Vector3dVector(line_set_points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                draw_color = [0, 1, 0] if draw_flow_lines_color == "green" else [0, 0, 1]
                line_set.colors = o3d.utility.Vector3dVector(
                    [draw_color for _ in range(len(lines))]
                )
                vis.add_geometry(line_set, reset_bounding_box=reset_view)


draw_frames(reset_view=True)

vis.run()
