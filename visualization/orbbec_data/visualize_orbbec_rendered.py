from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame
from bucketed_scene_flow_eval.utils import load_json

from pathlib import Path
import argparse
import open3d as o3d
import numpy as np
import json

from visualization.vis_lib import BaseCallbackVisualizer


class RawVisualizer(BaseCallbackVisualizer):

    def __init__(
        self,
        sequence: list[tuple[o3d.geometry.PointCloud, np.ndarray]],
        point_size: float = 0.1,
        line_width: float = 1.0,
        add_world_frame: bool = True,
    ):
        super().__init__(
            point_size=point_size, line_width=line_width, add_world_frame=add_world_frame
        )
        for pc, flow in sequence:
            self.add_geometry(pc)
            pc_points = np.asarray(pc.points)
            flowed_pc_points = pc_points + flow
            self.add_lineset(pc_points, flowed_pc_points, color=(1, 0, 0))

    def _register_callbacks(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        vis.register_key_callback(ord("S"), self.save_screenshot)
        vis.register_key_callback(ord("C"), self.save_camera_pose)
        vis.register_key_callback(ord("V"), self.load_camera_pose)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Path to the sequence folder
    parser.add_argument("sequence_folder", type=Path)
    parser.add_argument("--point_size", type=float, default=3)
    parser.add_argument("--camera_pose_file", type=Path, default=None)

    args = parser.parse_args()
    return args


def flow_folder_to_method_name(flow_folder: Path | None) -> str:
    if flow_folder is None:
        return "None"

    relevant_folder = flow_folder

    skip_strings = ["sequence_len", "LoaderType"]
    while any(skip_string in relevant_folder.name for skip_string in skip_strings):
        relevant_folder = relevant_folder.parent

    return relevant_folder.name


def load_sequence(sequence_data_folder: Path) -> list[tuple[o3d.geometry.PointCloud, np.ndarray]]:

    # Load the ply files
    ply_files = sorted(sequence_data_folder.glob("0*.ply"))
    flow_files = sorted(sequence_data_folder.glob("0*.json"))
    assert len(ply_files) > 0, f"No ply files found in {sequence_data_folder}"
    assert len(flow_files) > 0, f"No flow files found in {sequence_data_folder}"
    assert len(ply_files) == len(
        flow_files
    ), f"Number of ply files ({len(ply_files)}) does not match number of flow files ({len(flow_files)})"

    # Load the point clouds
    pcs = [o3d.io.read_point_cloud(str(ply_file)) for ply_file in ply_files]
    flows = [np.array(load_json(flow_file, verbose=False)) for flow_file in flow_files]
    return list(zip(pcs, flows))


def main():
    args = parse_args()

    sequence_folder = args.sequence_folder
    point_size = args.point_size
    camera_pose_file = args.camera_pose_file

    sequence = load_sequence(sequence_folder)

    vis = RawVisualizer(sequence, point_size=point_size)
    vis.run(camera_pose_path=camera_pose_file)


if __name__ == "__main__":
    main()
