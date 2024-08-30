import argparse
from pathlib import Path
import torch
import numpy as np

from models.whole_batch_optimization.checkpointing.model_loader import OptimCheckpointModelLoader
from dataloaders import TorchFullFrameInputSequence
from bucketed_scene_flow_eval.datastructures import (
    O3DVisualizer,
    PointCloud,
    TimeSyncedSceneFlowFrame,
    SupervisedPointCloudFrame,
    ColoredSupervisedPointCloudFrame,
)
from bucketed_scene_flow_eval.interfaces import AbstractDataset
from visualization.vis_lib import BaseCallbackVisualizer
from bucketed_scene_flow_eval.utils import load_json
from dataclasses import dataclass
from models.mini_batch_optimization import GigachadNSFModel
from models.components.neural_reps import ModelFlowResult, ModelOccFlowResult, QueryDirection
import open3d as o3d


class TrajectoryVisualizer(BaseCallbackVisualizer):

    def __init__(self, sequence_id: str, point_size: float = 0.1, line_width: float = 1.0):
        self.sequence_id = sequence_id
        super().__init__(point_size=point_size, line_width=line_width)

    def _get_screenshot_path(self) -> Path:
        return self.screenshot_path / self.sequence_id / "trajectory_screenshot.png"


@dataclass
class TrajectoryProblem:
    start_idx: int
    end_idx: int
    position: tuple[float, float, float]

    @staticmethod
    def from_json_file(json_file: Path) -> "TrajectoryProblem":
        json_data = load_json(json_file)
        return TrajectoryProblem(
            start_idx=json_data["start_idx"],
            end_idx=json_data["end_idx"],
            position=tuple(json_data["position"]),
        )

    def __post_init__(self):
        assert self.start_idx < self.end_idx
        assert isinstance(self.position, tuple)
        assert len(self.position) == 3


def visualize(
    model: GigachadNSFModel,
    full_sequence: TorchFullFrameInputSequence,
    base_dataset: AbstractDataset,
    trajectory_problem: TrajectoryProblem,
    camera_pose: Path,
):
    base_dataset_full_sequence = base_dataset[full_sequence.sequence_idx]
    vis = TrajectoryVisualizer(sequence_id="", point_size=3.0)
    vis.geometry_list.clear()
    color_list = vis._frame_list_to_color_list(len(full_sequence), "zebra")

    # Visualize all the frames
    for idx in range(trajectory_problem.start_idx, trajectory_problem.end_idx):
        scene_flow_frame = base_dataset_full_sequence[idx]
        color = color_list[idx]

        pc_frame: ColoredSupervisedPointCloudFrame = scene_flow_frame.pc
        assert isinstance(
            pc_frame, ColoredSupervisedPointCloudFrame
        ), f"Expected ColoredSupervisedPointCloudFrame, got {type(pc_frame)}"
        pc = pc_frame.full_global_pc
        colors = pc_frame.colors
        pc = full_sequence.get_full_global_pc(idx)
        vis.add_pointcloud(PointCloud(pc.cpu().numpy()), color=colors)

    query_positions = [np.array(trajectory_problem.position)]

    print(f"Querying {trajectory_problem.end_idx - trajectory_problem.start_idx} positions")
    for idx in range(trajectory_problem.start_idx, trajectory_problem.end_idx):
        last_position = query_positions[-1]
        last_position_torch = (
            torch.from_numpy(np.array([last_position])).float().to(full_sequence.device)
        )
        query_result: ModelFlowResult = model.model(
            last_position_torch,
            idx,
            len(full_sequence),
            QueryDirection.FORWARD,
        )
        next_position = last_position + query_result.flow[0].detach().cpu().numpy()
        query_positions.append(next_position)

    print(f"Plotting {len(query_positions)} positions")
    vis.add_sphere(query_positions[0], radius=0.1, color=(1, 0, 0))
    vis.add_trajectory(query_positions, color=(0, 0, 1))

    vis.run(camera_pose_path=camera_pose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint_file", type=Path)
    parser.add_argument("trajectory_problem", type=Path)
    parser.add_argument("camera_pose", type=Path)
    args = parser.parse_args()

    trajector_problem = TrajectoryProblem.from_json_file(args.trajectory_problem)
    model_loader = OptimCheckpointModelLoader.from_checkpoint(args.config, args.checkpoint_file)
    model, full_sequence, base_dataset = model_loader.load_model()
    model: GigachadNSFModel

    visualize(model, full_sequence, base_dataset, trajector_problem, args.camera_pose)


if __name__ == "__main__":
    main()
