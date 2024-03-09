import torch
import pandas as pd
import open3d as o3d
from bucketed_scene_flow_eval.datasets.shared_datastructures import AbstractSequence, SceneFlowItem
from bucketed_scene_flow_eval.datasets.argoverse2 import (
    ArgoverseSceneFlowSequenceLoader,
    ArgoverseSceneFlowSequence,
)
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
    "--flow_folders",
    type=str,
    nargs="+",  # This allows the user to input multiple flow folder paths
    default=["/efs/argoverse2/val_sceneflow_feather/"],
    help="Path(s) to the flow folder(s). This can be the ground truth flow, or dumped results from a model. Multiple paths can be provided.",
)
parser.add_argument("--point_size", type=float, default=3)
parser.add_argument("--frame_idx_step_size", type=int, default=1)
args = parser.parse_args()

sequence_id = args.sequence_id
print("Sequence ID: ", sequence_id)

flow_folders = [Path(flow_folder) for flow_folder in args.flow_folders]
for flow_folder in flow_folders:
    assert flow_folder.exists(), f"Flow folder {flow_folder} does not exist."

sequence_loaders = [
    ArgoverseSceneFlowSequenceLoader(
        args.sequence_folder,
        flow_folder,
        use_gt_flow=False,
    )
    for flow_folder in flow_folders
]
sequences = [sequence_loader.load_sequence(sequence_id) for sequence_loader in sequence_loaders]
print(f"Loaded {len(sequences)} sequences.")


class LazyFrameMatrix:
    def __init__(self, sequences: list[ArgoverseSceneFlowSequence]):
        self.sequences = sequences

    @property
    def shape(self):
        return len(self.sequences), len(self.sequences[0])

    def __getitem__(self, idx_tuple: tuple[int, int]) -> list[SceneFlowItem]:
        sequence_idx, frame_idx = idx_tuple
        sequence = self.sequences[sequence_idx]
        return [
            sequence.load(frame_idx, relative_to_idx=0, with_flow=True),
            sequence.load(frame_idx + 1, relative_to_idx=0, with_flow=False),
        ]


# Ensure all the sequences have the same length
for sequence in sequences[1:]:
    assert len(sequence) == len(
        sequences[0]
    ), f"All sequences must have the same length. Got {len(sequence)} and {len(sequences[0])}."

full_frame_matrix = LazyFrameMatrix(sequences)

draw_flow_lines_color = "red"
sequence_idx = 0
frame_starter_idx = args.sequence_start_idx


screenshot_path = Path() / "screenshots"
screenshot_path.mkdir(exist_ok=True)
# make open3d visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.get_render_option().point_size = args.point_size
vis.get_render_option().background_color = (1, 1, 1)
# vis.get_render_option().show_coordinate_frame = True
# set up vector
vis.get_view_control().set_up([0, 0, 1])


def get_flow_folder_name(sequence_idx: int) -> str:
    flow_folder = flow_folders[sequence_idx]
    name = flow_folder.name
    if name == "submission":
        return flow_folder.parent.name
    return name


def increase_sequence_idx(vis):
    global sequence_idx
    sequence_idx += 1
    if sequence_idx >= len(sequences):
        sequence_idx = 0
    print("Sequence:", sequence_idx, get_flow_folder_name(sequence_idx))
    vis.clear_geometries()
    draw_frames(reset_view=False)


def decrease_sequence_idx(vis):
    global sequence_idx
    sequence_idx -= 1
    if sequence_idx < 0:
        sequence_idx = len(sequences) - 1
    print("Sequence: ", sequence_idx, get_flow_folder_name(sequence_idx))
    vis.clear_geometries()
    draw_frames(reset_view=False)


def increase_starter_idx(vis):
    global frame_starter_idx
    frame_starter_idx += args.frame_idx_step_size
    if frame_starter_idx >= full_frame_matrix.shape[1] - 1:
        frame_starter_idx = 0
    print("Index: ", frame_starter_idx)
    vis.clear_geometries()
    draw_frames(reset_view=False)


def decrease_starter_idx(vis):
    global frame_starter_idx
    frame_starter_idx -= args.frame_idx_step_size
    if frame_starter_idx < 0:
        frame_starter_idx = full_frame_matrix.shape[1] - 2
    print("Index: ", frame_starter_idx)
    vis.clear_geometries()
    draw_frames(reset_view=False)


def toggle_flow_lines(vis):
    global draw_flow_lines_color

    if draw_flow_lines_color is None:
        draw_flow_lines_color = "red"
    elif draw_flow_lines_color == "red":
        draw_flow_lines_color = "green"
    elif draw_flow_lines_color == "green":
        draw_flow_lines_color = "blue"
    elif draw_flow_lines_color == "blue":
        draw_flow_lines_color = None
    else:
        raise ValueError(f"Invalid draw_flow_lines_color: {draw_flow_lines_color}")
    vis.clear_geometries()
    draw_frames(reset_view=False)


def color_name_to_rgb(color_name: str) -> tuple[int, int, int]:
    if color_name == "red":
        return (1, 0, 0)
    elif color_name == "green":
        return (0, 1, 0)
    elif color_name == "blue":
        return (0, 0, 1)
    else:
        raise ValueError(f"Invalid color_name: {color_name}")


def save_screenshot(vis):
    save_name = (
        screenshot_path
        / sequence_id
        / f"{frame_starter_idx:06d}_{get_flow_folder_name(sequence_idx)}.png"
    )
    save_name.parent.mkdir(exist_ok=True, parents=True)
    vis.capture_screen_image(str(save_name))


vis.register_key_callback(ord("F"), toggle_flow_lines)
# left arrow decrease starter_idx
vis.register_key_callback(263, decrease_starter_idx)
# right arrow increase starter_idx
vis.register_key_callback(262, increase_starter_idx)
# up arrow increase sequence_idx
vis.register_key_callback(265, increase_sequence_idx)
# down arrow decrease sequence_idx
vis.register_key_callback(264, decrease_sequence_idx)
vis.register_key_callback(ord("S"), save_screenshot)

print("#############################################################")
print("Flow moves from the gray point cloud to the white point cloud\n")
print("Press F to toggle flow lines")
print("Press left or right arrow to change starter_idx")
print(f"Press S to save screenshot (saved to {screenshot_path.absolute()})")
print("#############################################################")


def draw_frames(reset_view=False):

    frame_list = full_frame_matrix[sequence_idx, frame_starter_idx]
    assert len(frame_list) == 2, f"Expected 2 frames, but got {len(frame_list)}"
    color_list = [(0, 0, 1), (0, 1, 0)]

    def _colorize_pc(pc: o3d.geometry.PointCloud, color_tuple: tuple[float, float, float]):
        pc_color = np.ones_like(pc.points) * np.array(color_tuple)
        return o3d.utility.Vector3dVector(pc_color)

    for idx, frame_dict in enumerate(frame_list):
        pc = frame_dict.relative_pc
        pose = frame_dict.relative_pose
        flowed_pc = frame_dict.relative_flowed_pc

        # Add base point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)
        pcd.colors = _colorize_pc(pcd, color_list[idx])
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
                draw_color = color_name_to_rgb(draw_flow_lines_color)
                line_set.colors = o3d.utility.Vector3dVector(
                    [draw_color for _ in range(len(lines))]
                )
                vis.add_geometry(line_set, reset_bounding_box=reset_view)


draw_frames(reset_view=True)

vis.run()
