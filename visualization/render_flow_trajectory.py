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
from bucketed_scene_flow_eval.utils import load_json, save_json
from dataclasses import dataclass
from models.mini_batch_optimization import GigachadNSFModel
from models.components.neural_reps import ModelFlowResult, ModelOccFlowResult, QueryDirection
import open3d as o3d
import json
import tqdm
import multiprocessing as mp


@dataclass
class TrajectoryProblem:
    start_idx: int
    end_idx: int
    positions: list[tuple[float, float, float]]

    @staticmethod
    def from_json_file(json_file: Path) -> "TrajectoryProblem":
        json_data = load_json(json_file)
        return TrajectoryProblem(
            start_idx=json_data["start_idx"],
            end_idx=json_data["end_idx"],
            positions=[tuple(e) for e in json_data["positions"]],
        )

    def __post_init__(self):
        assert self.start_idx < self.end_idx
        assert len(self.positions) > 0
        for position in self.positions:
            assert isinstance(position, tuple)
            assert len(position) == 3


def save_json_truncated(filename: Path, data):
    print(f"Saving to {filename}")
    with open(filename, "w") as json_file:
        raw_str = json.dumps(
            json.loads(
                json.dumps(data, default=str),
                parse_float=lambda x: round(float(x), 3),
            )
        )
        json_file.write(raw_str)


@dataclass
class TrajectoryData:
    trajectories: list[list[tuple[float, float, float]]]

    def save(self, parent_folder: Path, num_steps: int = 1):
        parent_folder.mkdir(parents=True, exist_ok=True)
        json_path = parent_folder / f"trajectories_num_substeps_{num_steps}.json"
        save_json_truncated(json_path, self.trajectories)


@dataclass
class SceneFlowData:
    points: np.ndarray
    colors: np.ndarray
    flows: np.ndarray

    def __post_init__(self):
        assert (
            self.points.shape[0] == self.colors.shape[0]
        ), f"{self.points.shape} != {self.colors.shape}"
        assert (
            self.points.shape[0] == self.flows.shape[0]
        ), f"{self.points.shape} != {self.flows.shape}"

    def save(self, parent_folder: Path, idx: int):
        parent_folder.mkdir(parents=True, exist_ok=True)
        # PLY file for the point cloud
        ply_path = parent_folder / f"{idx:04d}.ply"
        with open(ply_path, "w") as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {self.points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            # Write point data
            for point, color in zip(self.points, self.colors):
                x, y, z = point
                r, g, b = color
                f.write(f"{x:.2f} {y:.2f} {z:.2f} {int(r*255)} {int(g*255)} {int(b*255)}\n")

        # JSON file for the flow
        json_path = parent_folder / f"{idx:04d}.json"
        save_json_truncated(json_path, self.flows.tolist())


def save_result(result: SceneFlowData, parent_folder: Path, idx: int):
    result.save(parent_folder, idx)


def render_flows(
    model: GigachadNSFModel,
    full_sequence: TorchFullFrameInputSequence,
    base_dataset: AbstractDataset,
    trajectory_problem: TrajectoryProblem,
    output_folder: Path,
) -> list[SceneFlowData]:
    base_dataset_full_sequence = base_dataset[full_sequence.sequence_idx]

    results: list[SceneFlowData] = []

    # Use torch inference mode
    model.model = model.model.eval()
    with torch.no_grad():
        for idx in tqdm.tqdm(
            range(trajectory_problem.start_idx, trajectory_problem.end_idx), desc="Rendering Flows"
        ):
            torch_query_points = full_sequence.get_global_pc(idx)
            scene_flow_frame = base_dataset_full_sequence[idx]

            query_result: ModelFlowResult = model.model(
                torch_query_points,
                idx,
                len(full_sequence),
                QueryDirection.FORWARD,
            )
            flow_np = query_result.flow.detach().cpu().numpy()

            pc_frame: SupervisedPointCloudFrame = scene_flow_frame.pc
            if isinstance(pc_frame, ColoredSupervisedPointCloudFrame):
                color_np = pc_frame.colors[pc_frame.mask]
            else:
                color_np = np.ones_like(flow_np)
            pc_np = pc_frame.global_pc.points
            results.append(SceneFlowData(points=pc_np, colors=color_np, flows=flow_np))

    print("Saving results")
    arguments_lst = [(result, output_folder, idx) for idx, result in enumerate(results)]
    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(save_result, arguments_lst)
    return results


def render_trajectory_euler(
    model: GigachadNSFModel,
    full_sequence: TorchFullFrameInputSequence,
    trajectory_problem: TrajectoryProblem,
    output_folder: Path,
):
    trajectory_position_list = [[np.array(p)] for p in trajectory_problem.positions]

    print(
        f"Querying {trajectory_problem.end_idx - trajectory_problem.start_idx} positions from {trajectory_problem.start_idx} to {trajectory_problem.end_idx}"
    )
    for idx in tqdm.tqdm(
        range(trajectory_problem.start_idx, trajectory_problem.end_idx), desc="Rendering Trajectory"
    ):

        last_position_batch = np.array([e[-1] for e in trajectory_position_list])
        last_position_torch = (
            torch.from_numpy(np.array(last_position_batch)).float().to(full_sequence.device)
        )
        query_result: ModelFlowResult = model.model(
            last_position_torch,
            idx,
            len(full_sequence),
            QueryDirection.FORWARD,
        )
        flow_batch = query_result.flow.detach().cpu().numpy()

        for idx, flow in enumerate(flow_batch):
            next_position = trajectory_position_list[idx][-1] + flow
            trajectory_position_list[idx].append(next_position)

    trajectory_data = TrajectoryData(
        trajectories=[[tuple(e) for e in trajectory] for trajectory in trajectory_position_list]
    )
    trajectory_data.save(output_folder)


def render_trajectory_midpoint(
    model: GigachadNSFModel,
    full_sequence: TorchFullFrameInputSequence,
    trajectory_problem: TrajectoryProblem,
    output_folder: Path,
    num_substeps: int = 2,
):
    trajectory_position_list = [[np.array(p)] for p in trajectory_problem.positions]

    substep_range = list(
        np.arange(trajectory_problem.start_idx, trajectory_problem.end_idx, 1 / num_substeps)
    )

    print(
        f"Querying {len(substep_range)} positions from {trajectory_problem.start_idx} to {trajectory_problem.end_idx}"
    )

    full_sequence_length = len(full_sequence)
    for idx in tqdm.tqdm(
        substep_range,
        desc="Rendering Trajectory",
    ):
        idx = float(idx)

        last_position_batch = np.array([e[-1] for e in trajectory_position_list])
        last_position_torch = (
            torch.from_numpy(np.array(last_position_batch)).float().to(full_sequence.device)
        )
        query_result: ModelFlowResult = model.model(
            last_position_torch,
            idx,
            full_sequence_length,
            QueryDirection.FORWARD,
        )
        flow_batch = query_result.flow.detach().cpu().numpy()

        for idx, flow in enumerate(flow_batch):
            next_position = trajectory_position_list[idx][-1] + (flow / num_substeps)
            trajectory_position_list[idx].append(next_position)

    trajectory_data = TrajectoryData(
        trajectories=[[tuple(e) for e in trajectory] for trajectory in trajectory_position_list]
    )
    trajectory_data.save(output_folder, num_substeps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint_file", type=Path)
    parser.add_argument("trajectory_problem", type=Path)
    parser.add_argument("output_folder", type=Path)
    args = parser.parse_args()

    trajector_problem = TrajectoryProblem.from_json_file(args.trajectory_problem)
    model_loader = OptimCheckpointModelLoader.from_checkpoint(args.config, args.checkpoint_file)
    model, full_sequence, base_dataset = model_loader.load_model()
    model: GigachadNSFModel
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    render_flows(model, full_sequence, base_dataset, trajector_problem, output_folder)
    render_trajectory_euler(model, full_sequence, trajector_problem, output_folder)
    render_trajectory_midpoint(
        model, full_sequence, trajector_problem, output_folder, num_substeps=2
    )
    render_trajectory_midpoint(
        model, full_sequence, trajector_problem, output_folder, num_substeps=4
    )
    render_trajectory_midpoint(
        model, full_sequence, trajector_problem, output_folder, num_substeps=8
    )


if __name__ == "__main__":
    main()
