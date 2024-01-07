from bucketed_scene_flow_eval.datastructures import PointCloudFrame, RGBFrame, RawSceneSequence, QuerySceneSequence, SE3, GroundTruthParticleTrajectories, Timestamp
from bucketed_scene_flow_eval.datasets import *
from pathlib import Path
import torch
from typing import Tuple, Dict, List, Any
from pointclouds import to_fixed_array, from_fixed_array
import numpy as np
import time
from dataclasses import dataclass


@dataclass
class BucketedSceneFlowItem():
    dataset_log_id: str
    dataset_idx: int
    query_timestamp: Timestamp
    source_pc: torch.FloatTensor  # (N, 3)
    target_pc: torch.FloatTensor  # (M, 3)
    source_pose: SE3
    target_pose: SE3
    gt_flowed_source_pc: torch.FloatTensor  # (N, 3)
    gt_pc_class_mask: torch.LongTensor  # (N,)

    # Included for completeness, these are the full percepts provided by the
    # dataloader rather than just the scene flow query points.
    # This allows for the data loader to e.g. provide more point clouds for accumulation.
    full_percept_pcs_array_stack: torch.FloatTensor  # (K, PadN, 3)
    full_percept_pose_array_stack: torch.FloatTensor  # (K, 4, 4)
    gt_trajectories: GroundTruthParticleTrajectories

    def to(self, device):
        self.source_pc = self.source_pc.to(device)
        self.target_pc = self.target_pc.to(device)
        self.full_percept_pcs_array_stack = self.full_percept_pcs_array_stack.to(
            device)
        self.full_percept_pose_array_stack = self.full_percept_pose_array_stack.to(
            device)
        self.gt_flowed_source_pc = self.gt_flowed_source_pc.to(device)
        self.gt_pc_class_mask = self.gt_pc_class_mask.to(device)
        return self

    def __post_init__(self):
        assert self.source_pc.shape[
            1] == 3, f"Source PC has shape {self.source_pc.shape}"
        assert self.target_pc.shape[
            1] == 3, f"Target PC has shape {self.target_pc.shape}"
        assert self.gt_flowed_source_pc.shape[
            1] == 3, f"GT flowed source PC has shape {self.gt_flowed_source_pc.shape}"
        assert self.source_pc.shape == self.gt_flowed_source_pc.shape, f"Source PC {self.source_pc.shape} != GT flowed source PC {self.gt_flowed_source_pc.shape}"

        assert self.gt_pc_class_mask.shape[0] == self.gt_flowed_source_pc.shape[
            0], f"GT PC class mask {self.gt_pc_class_mask.shape} != GT flowed source PC {self.gt_flowed_source_pc.shape}"

        assert isinstance(self.source_pose, SE3), f"Source pose is not an SE3"
        assert isinstance(self.target_pose, SE3), f"Target pose is not an SE3"

        assert self.full_percept_pcs_array_stack.shape[
            2] == 3, f"Percept PC has shape {self.full_percept_pcs_array_stack.shape}, but should be (K, PadN, 3)"
        assert self.full_percept_pose_array_stack.shape[
            0] == self.full_percept_pcs_array_stack.shape[
                0], f"Full percept and pose stacks have different number of entries"

    def full_percept(self, idx: int) -> Tuple[np.ndarray, SE3]:
        return from_fixed_array(
            self.full_percept_pcs_array_stack[idx]), SE3.from_array(
                self.full_percept_pose_array_stack[idx])

    def full_percepts(self) -> List[Tuple[np.ndarray, SE3]]:
        return [
            self.full_percept(idx)
            for idx in range(len(self.full_percept_pcs_array_stack))
        ]


class BucketedSceneFlowDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_name: str,
                 root_dir: Path,
                 max_pc_points: int = 120000,
                 **kwargs):
        self.dataset = self._construct_dataset(
            dataset_name, dict(root_dir=root_dir, **kwargs))
        self.max_pc_points = max_pc_points

    def _construct_dataset(self, dataset_name: str, arguments: dict):
        dataset_name = dataset_name.strip().lower()
        if dataset_name == "argoverse2sceneflow":
            return Argoverse2SceneFlow(**arguments)

        raise ValueError(f"Unknown dataset name {dataset_name}")

    def __len__(self):
        return len(self.dataset)

    def evaluator(self):
        return self.dataset.evaluator()

    def collate_fn(self, batch: List[BucketedSceneFlowItem]):
        return batch

    def _process_query(
        self, query: QuerySceneSequence
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[SE3, SE3],
               List[np.ndarray], List[SE3]]:
        assert len(
            query.query_trajectory_timestamps
        ) == 2, f"Query {query} has more than two timestamps. Only Scene Flow problems are supported."
        scene = query.scene_sequence

        # These contain the full problem percepts, not just the ones in the query.
        full_percept_pc_arrays: List[np.ndarray] = []
        full_percept_poses: List[SE3] = []
        # These contain only the percepts in the query.0
        problem_pc_arrays: List[np.ndarray] = []
        problem_poses: List[SE3] = []

        for timestamp in scene.get_percept_timesteps():
            pc_frame = scene[timestamp].pc_frame
            pc_array = pc_frame.global_pc.points.astype(np.float32)
            pose = pc_frame.global_pose

            full_percept_pc_arrays.append(pc_array)
            full_percept_poses.append(pose)

            if timestamp in query.query_trajectory_timestamps:
                problem_pc_arrays.append(pc_array)
                problem_poses.append(pose)

        assert len(full_percept_pc_arrays) == len(
            full_percept_poses
        ), f"Percept arrays and poses have different lengths."
        assert len(problem_pc_arrays) == len(
            problem_poses), f"Percept arrays and poses have different lengths."
        assert len(problem_pc_arrays) == len(
            query.query_trajectory_timestamps
        ), f"Percept arrays and poses have different lengths."

        return problem_pc_arrays, problem_poses, full_percept_pc_arrays, full_percept_poses

    def _process_gt(self, result: GroundTruthParticleTrajectories):
        flowed_source_pc = result.world_points[:, 1].astype(np.float32)
        point_cls_array = result.cls_ids
        return flowed_source_pc, point_cls_array

    def __getitem__(self, idx):
        dataset_entry: Tuple[
            QuerySceneSequence,
            GroundTruthParticleTrajectories] = self.dataset[idx]
        query, gt_result = dataset_entry

        (source_pc, target_pc), (
            source_pose, target_pose
        ), full_pc_points_list, full_pc_poses_list = self._process_query(query)

        gt_flowed_source_pc, gt_point_classes = self._process_gt(gt_result)

        full_percept_pcs_array_stack = np.stack([
            to_fixed_array(pc, self.max_pc_points)
            for pc in full_pc_points_list
        ],
                                                axis=0)
        full_percept_pose_array_stack = np.stack(
            [pose.to_array() for pose in full_pc_poses_list], axis=0)

        item = BucketedSceneFlowItem(
            dataset_log_id=query.scene_sequence.log_id,
            dataset_idx=idx,
            query_timestamp=query.query_particles.query_init_timestamp,
            source_pc=torch.from_numpy(source_pc),
            target_pc=torch.from_numpy(target_pc),
            source_pose=source_pose,
            target_pose=target_pose,
            full_percept_pcs_array_stack=torch.from_numpy(
                full_percept_pcs_array_stack),
            full_percept_pose_array_stack=torch.from_numpy(
                full_percept_pose_array_stack),
            gt_flowed_source_pc=torch.from_numpy(gt_flowed_source_pc),
            gt_pc_class_mask=torch.from_numpy(gt_point_classes),
            gt_trajectories=gt_result)

        return item
