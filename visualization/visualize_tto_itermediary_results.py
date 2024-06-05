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
class SubSubsequenceInfo:
    start_idx: int
    size: int

    def __repr__(self) -> str:
        return f"Start: {self.start_idx} Size: {self.size}"


@dataclass(kw_only=True)
class VisState:
    flow_color: ColorEnum = ColorEnum.RED
    intermedary_result_idx: int = 0
    subsubsequence: SubSubsequenceInfo | None = None

    def __repr__(self) -> str:
        ret_str = f"Vis State: Opt Step {self.intermedary_result_idx}"
        if self.subsubsequence is not None:
            ret_str += f" SubSubsequence: ({self.subsubsequence})"
        return ret_str


class ResultsVisualizer(BaseCallbackVisualizer):

    def __init__(
        self,
        dataset: AbstractDataset,
        intermediary_results_folder: Path,
        subsequence_length: int,
        subsubsequence_length: int | None,
    ):

        self.dataset = dataset
        self.intermediary_results_folder = intermediary_results_folder
        self._load_intermediary_flows()
        self.subsequence_length = subsequence_length

        self.vis_state = VisState(intermedary_result_idx=len(self.intermediary_files) - 1)
        if subsubsequence_length is not None:
            self.vis_state.subsubsequence = SubSubsequenceInfo(
                start_idx=0,
                size=subsubsequence_length,
            )

        super().__init__(
            screenshot_path=Path()
            / "screenshots_intermediary"
            / f"dataset_idx_{self.dataset_idx:010d}"
        )

    def _load_intermediary_flows(self):
        """
        Load / update a cache of all existing intermediary results.

        This is called whenever the user changes the intermediary result index, in order to support
        live updates of the visualizer from a running method.
        """
        intermediary_results_folder = self.intermediary_results_folder
        assert intermediary_results_folder.exists(), f"{intermediary_results_folder} does not exist"
        intermediary_files = sorted(intermediary_results_folder.glob("*.pkl"))
        assert (
            len(intermediary_files) > 0
        ), f"No intermediary files found in {intermediary_results_folder}"

        self.dataset_idx = 0
        self.intermediary_files = intermediary_files

    def _load_ego_lidar(self) -> list[EgoLidarFlow]:
        raw_intermediary_flows: list[tuple[np.ndarray, np.ndarray]] = load_pickle(
            self.intermediary_files[self.vis_state.intermedary_result_idx], verbose=False
        )
        return [EgoLidarFlow(*raw_flow) for raw_flow in raw_intermediary_flows]

    def _get_result_data(self) -> list[TimeSyncedSceneFlowFrame]:
        """
        Loads the dataset frame list and adds the intermediary ego flow to each frame.

        Caches the dataset frame list and intermediary ego flows if the dataset index has not changed.
        """

        # Check to see if self._cached_dataset_idx is the same as self.dataset_idx.
        # The field will not exist if it has not been set yet.
        if hasattr(self, "_cached_dataset_idx") and self._cached_dataset_idx == self.dataset_idx:
            assert hasattr(
                self, "_cached_dataset_frame_list"
            ), "Cached dataset frame list not found"
            return self._cached_dataset_frame_list

        assert len(self.dataset) > 0, "No sequences found in dataset"
        print(f"Datset has {len(self.dataset)} sequences")
        dataset_frame_list = self.dataset[0]
        intermediary_ego_flows = self._load_ego_lidar()

        assert (
            len(dataset_frame_list) == len(intermediary_ego_flows) + 1
        ), f"Expected one more frame in dataset than intermediary flows; instead found {len(dataset_frame_list)} and {len(intermediary_ego_flows)}"

        for frame_info, ego_flow in zip(dataset_frame_list, intermediary_ego_flows):
            # Add the intermediary ego flow to the frame info
            frame_info.flow = ego_flow

        self._cached_dataset_idx = self.dataset_idx
        self._cached_dataset_frame_list = dataset_frame_list

        return dataset_frame_list

    def _get_screenshot_path(self) -> Path:
        file_stem = f"{self.vis_state.intermedary_result_idx:08d}"
        if self.vis_state.subsubsequence is not None:
            file_stem += f"_subsubsequence_{self.vis_state.subsubsequence.start_idx:08d}"
        return self.screenshot_path / f"{file_stem}.png"

    def _print_instructions(self):
        print("#############################################################")
        print("Flow moves from the gray point cloud to the white point cloud\n")
        print("Press up or down arrow to change intermediary result")
        print(f"Press S to save screenshot (saved to {self.screenshot_path.absolute()})")
        print("Press E to jump to end of sequence")
        print("#############################################################")

    def _register_callbacks(self, vis: o3d.visualization.VisualizerWithKeyCallback):
        super()._register_callbacks(vis)
        # up arrow increase result_idx
        vis.register_key_callback(265, self.increase_result_idx)
        # down arrow decrease result_idx
        vis.register_key_callback(264, self.decrease_result_idx)
        # right arrow increase subsubsequence_idx
        vis.register_key_callback(262, self.increase_subsubsequence_idx)
        # left arrow decrease subsubsequence_idx
        vis.register_key_callback(263, self.decrease_subsubsequence_idx)
        # E to jump to end of sequence
        vis.register_key_callback(ord("E"), self.jump_to_end_of_sequence)

    def increase_result_idx(self, vis):
        self._load_intermediary_flows()  # Refreshj cache to support live updates
        self.vis_state.intermedary_result_idx += 1
        if self.vis_state.intermedary_result_idx >= len(self.intermediary_files):
            self.vis_state.intermedary_result_idx = 0
        self.draw_everything(vis, reset_view=False)

    def decrease_result_idx(self, vis):
        self._load_intermediary_flows()  # Refreshj cache to support live updates
        self.vis_state.intermedary_result_idx -= 1
        if self.vis_state.intermedary_result_idx < 0:
            self.vis_state.intermedary_result_idx = len(self.intermediary_files) - 1
        self.draw_everything(vis, reset_view=False)

    def increase_subsubsequence_idx(self, vis):
        if self.vis_state.subsubsequence is None:
            return
        self.vis_state.subsubsequence.start_idx += 1
        end_idx = self.vis_state.subsubsequence.start_idx + self.vis_state.subsubsequence.size
        if end_idx >= self.subsequence_length:
            self.vis_state.subsubsequence.start_idx = 0
        self.draw_everything(vis, reset_view=False)

    def decrease_subsubsequence_idx(self, vis):
        if self.vis_state.subsubsequence is None:
            return
        self.vis_state.subsubsequence.start_idx -= 1
        if self.vis_state.subsubsequence.start_idx < 0:
            self.vis_state.subsubsequence.start_idx = (
                self.subsequence_length - self.vis_state.subsubsequence.size
            )
        self.draw_everything(vis, reset_view=False)

    def jump_to_end_of_sequence(self, vis):
        self._load_intermediary_flows()
        self.vis_state.intermedary_result_idx = len(self.intermediary_files) - 1
        self.draw_everything(vis, reset_view=False)

    def draw_everything(self, vis, reset_view=False):
        self.geometry_list.clear()
        print(f"Vis State: {self.vis_state}")
        frame_list = self._get_result_data()
        color_list = self._frame_list_to_color_list(len(frame_list))

        if self.vis_state.subsubsequence is not None:
            start_idx = self.vis_state.subsubsequence.start_idx
            size = self.vis_state.subsubsequence.size
            frame_list = frame_list[start_idx : start_idx + size]
            color_list = color_list[start_idx : start_idx + size]

        for idx, (flow_frame, color) in enumerate(zip(frame_list, color_list)):
            pc = flow_frame.pc.global_pc
            self.add_pointcloud(pc, color=color)
            flowed_pc = flow_frame.pc.flow(flow_frame.flow).global_pc

            draw_color = self.vis_state.flow_color.rgb

            # Add flowed point cloud
            if (draw_color is not None) and (flowed_pc is not None) and (idx < len(frame_list) - 1):
                self.add_lineset(pc, flowed_pc, color=draw_color)
        vis.clear_geometries()
        self.render(vis, reset_view=reset_view)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="Argoverse2NonCausalSceneFlow")
    parser.add_argument("root_dir", type=Path)
    parser.add_argument("subsequence_length", type=int)
    parser.add_argument("--subsubsequence_length", type=int, default=-1)
    parser.add_argument("--subsequence_id", type=str, required=True)
    parser.add_argument("intermediary_results_folder", type=Path)
    args = parser.parse_args()

    subsequence_length = args.subsequence_length

    dataset = construct_dataset(
        name=args.dataset_name,
        args=dict(
            root_dir=args.root_dir,
            subsequence_length=subsequence_length,
            with_ground=False,
            range_crop_type="ego",
            log_subset=[args.subsequence_id] if args.subsequence_id is not None else None,
        ),
    )
    assert len(dataset) > 0, "No sequences found in dataset"
    print(f"Datset has {len(dataset)} sequences")
    subsubsequence_length = args.subsubsequence_length
    if subsubsequence_length < 0:
        subsubsequence_length = None
    intermediary_results_folder = args.intermediary_results_folder
    # If the results folder is a pkl file, then grab the parent directory
    if intermediary_results_folder.is_file():
        intermediary_results_folder = intermediary_results_folder.parent
    visualizer = ResultsVisualizer(
        dataset, intermediary_results_folder, subsequence_length, subsubsequence_length
    )

    visualizer.run()


if __name__ == "__main__":
    main()
