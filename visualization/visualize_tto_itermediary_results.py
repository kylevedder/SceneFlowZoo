import argparse
from pathlib import Path
from bucketed_scene_flow_eval.utils import load_pickle
from bucketed_scene_flow_eval.datastructures import (
    O3DVisualizer,
    EgoLidarFlow,
    PointCloud,
    TimeSyncedSceneFlowFrame,
)
from bucketed_scene_flow_eval.datasets import (
    construct_dataset,
    AbstractDataset,
)
from visualization.vis_lib import ColorEnum, BaseCallbackVisualizer
from dataclasses import dataclass
import numpy as np
import open3d as o3d


@dataclass(kw_only=True)
class VisState:
    flow_color: ColorEnum = ColorEnum.RED
    intermedary_result_idx: int = 0


class ResultsVisualizer(BaseCallbackVisualizer):

    def __init__(self, dataset: AbstractDataset, intermediary_results_folder: Path):

        self.vis_state = VisState()
        self.dataset = dataset
        self.intermediary_results_folder = intermediary_results_folder
        self._load_intermediary_flows()

        super().__init__(
            screenshot_path=Path()
            / "screenshots_intermediary"
            / f"dataset_idx_{self.dataset_idx:010d}"
        )

    def _load_intermediary_flows(self):
        intermediary_results_folder = self.intermediary_results_folder
        dataset_idx = int(intermediary_results_folder.name.split("_")[-1])
        assert intermediary_results_folder.exists(), f"{intermediary_results_folder} does not exist"
        intermediary_files = sorted(intermediary_results_folder.glob("*.pkl"))
        assert (
            len(intermediary_files) > 0
        ), f"No intermediary files found in {intermediary_results_folder}"

        print(f"Found {len(intermediary_files)} intermediary files for dataset idx {dataset_idx}")
        self.dataset_idx, self.intermediary_files = dataset_idx, intermediary_files

    def _load_ego_lidar(self) -> list[EgoLidarFlow]:
        raw_intermediary_flows: list[tuple[np.ndarray, np.ndarray]] = load_pickle(
            self.intermediary_files[self.vis_state.intermedary_result_idx], verbose=False
        )
        return [EgoLidarFlow(*raw_flow) for raw_flow in raw_intermediary_flows]

    def _get_result_data(self) -> list[TimeSyncedSceneFlowFrame]:
        dataset_frame_list = self.dataset[self.dataset_idx]
        intermediary_ego_flows = self._load_ego_lidar()

        assert (
            len(dataset_frame_list) == len(intermediary_ego_flows) + 1
        ), f"Expected one more frame in dataset than intermediary flows; instead found {len(dataset_frame_list)} and {len(intermediary_ego_flows)}"

        for frame_info, ego_flow in zip(dataset_frame_list, intermediary_ego_flows):
            # Add the intermediary ego flow to the frame info
            frame_info.flow = ego_flow

        return dataset_frame_list

    def _get_screenshot_path(self) -> Path:
        return self.screenshot_path / f"{self.vis_state.intermedary_result_idx:08d}.png"

    def _print_instructions(self):
        print("#############################################################")
        print("Flow moves from the gray point cloud to the white point cloud\n")
        print("Press up or down arrow to change intermediary result")
        print(f"Press S to save screenshot (saved to {self.screenshot_path.absolute()})")
        print("Press E to jump to end of sequence")
        print("#############################################################")

    def _register_callbacks(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        super()._register_callbacks(vis)
        # up arrow increase sequence_idx
        vis.register_key_callback(265, self.increase_sequence_idx)
        # down arrow decrease sequence_idx
        vis.register_key_callback(264, self.decrease_sequence_idx)
        # E to jump to end of sequence
        vis.register_key_callback(ord("E"), self.jump_to_end_of_sequence)

    def increase_sequence_idx(self, vis):
        self._load_intermediary_flows()
        self.vis_state.intermedary_result_idx += 1
        if self.vis_state.intermedary_result_idx >= len(self.intermediary_files):
            self.vis_state.intermedary_result_idx = 0
        self.draw_everything(vis, reset_view=False)

    def decrease_sequence_idx(self, vis):
        self._load_intermediary_flows()
        self.vis_state.intermedary_result_idx -= 1
        if self.vis_state.intermedary_result_idx < 0:
            self.vis_state.intermedary_result_idx = len(self.intermediary_files) - 1
        self.draw_everything(vis, reset_view=False)

    def jump_to_end_of_sequence(self, vis):
        self._load_intermediary_flows()
        self.vis_state.intermedary_result_idx = len(self.intermediary_files) - 1
        self.draw_everything(vis, reset_view=False)

    def draw_everything(self, vis, reset_view=False):
        self.geometry_list.clear()
        print(f"Vis State: {self.intermediary_files[self.vis_state.intermedary_result_idx].stem}")
        frame_list = self._get_result_data()
        color_list = self._frame_list_to_color_list(len(frame_list))

        for idx, flow_frame in enumerate(frame_list):
            pc = flow_frame.pc.global_pc
            self.add_pointcloud(pc, color=color_list[idx])
            flowed_pc = flow_frame.pc.flow(flow_frame.flow).global_pc

            draw_color = self.vis_state.flow_color.rgb

            # Add flowed point cloud
            if (draw_color is not None) and (flowed_pc is not None) and (idx < len(frame_list) - 1):
                self.add_lineset(pc, flowed_pc, color=draw_color)
        vis.clear_geometries()
        self.render(vis, reset_view=reset_view)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Argoverse2CausalSceneFlow")
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("subsequence_length", type=int)
    parser.add_argument("intermediary_results_folder", type=Path)
    args = parser.parse_args()

    dataset = construct_dataset(
        name=args.dataset_name,
        args=dict(
            root_dir=args.root_dir, subsequence_length=args.subsequence_length, with_ground=False
        ),
    )
    intermediary_results_folder = args.intermediary_results_folder
    # If the results folder is a pkl file, then grab the parent directory
    if intermediary_results_folder.is_file():
        intermediary_results_folder = intermediary_results_folder.parent
    visualizer = ResultsVisualizer(dataset, intermediary_results_folder)

    visualizer.run()


if __name__ == "__main__":
    main()
