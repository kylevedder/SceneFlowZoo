from .dataclasses import BucketedSceneFlowItem
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from bucketed_scene_flow_eval.datasets import *
from bucketed_scene_flow_eval.datastructures import (
    SE3,
    GroundTruthPointFlow,
    QuerySceneSequence,
)

from pointclouds import to_fixed_array


class BucketedSceneFlowDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, root_dir: Path, max_pc_points: int = 120000, **kwargs):
        self.dataset = self._construct_dataset(dataset_name, dict(root_dir=root_dir, **kwargs))
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
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[SE3, SE3], List[np.ndarray], List[SE3]]:
        assert (
            len(query.query_flow_timestamps) == 2
        ), f"Query {query} has more than two timestamps. Only Scene Flow problems are supported."
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

            if timestamp in query.query_flow_timestamps:
                problem_pc_arrays.append(pc_array)
                problem_poses.append(pose)

        assert len(full_percept_pc_arrays) == len(
            full_percept_poses
        ), f"Percept arrays and poses have different lengths."
        assert len(problem_pc_arrays) == len(
            problem_poses
        ), f"Percept arrays and poses have different lengths."
        assert len(problem_pc_arrays) == len(
            query.query_flow_timestamps
        ), f"Percept arrays and poses have different lengths."

        return problem_pc_arrays, problem_poses, full_percept_pc_arrays, full_percept_poses

    def _process_gt(self, result: GroundTruthPointFlow):
        flowed_source_pc = result.world_points[:, 1].astype(np.float32)
        point_cls_array = result.cls_ids
        return flowed_source_pc, point_cls_array

    def __getitem__(self, idx):
        dataset_entry: Tuple[QuerySceneSequence, GroundTruthPointFlow] = self.dataset[
            idx
        ]
        query, gt_result = dataset_entry

        (
            (source_pc, target_pc),
            (source_pose, target_pose),
            full_pc_points_list,
            full_pc_poses_list,
        ) = self._process_query(query)

        gt_flowed_source_pc, gt_point_classes = self._process_gt(gt_result)

        full_percept_pcs_array_stack = np.stack(
            [to_fixed_array(pc, self.max_pc_points) for pc in full_pc_points_list], axis=0
        )
        full_percept_pose_array_stack = np.stack(
            [pose.to_array() for pose in full_pc_poses_list], axis=0
        )
        item = BucketedSceneFlowItem(
            dataset_log_id=query.scene_sequence.log_id,
            dataset_idx=idx,
            query_timestamp=query.query_particles.query_init_timestamp,
            source_pc=torch.from_numpy(source_pc),
            target_pc=torch.from_numpy(target_pc),
            source_pose=source_pose,
            target_pose=target_pose,
            full_percept_pcs_array_stack=torch.from_numpy(full_percept_pcs_array_stack),
            full_percept_pose_array_stack=torch.from_numpy(full_percept_pose_array_stack),
            gt_flowed_source_pc=torch.from_numpy(gt_flowed_source_pc),
            gt_pc_class_mask=torch.from_numpy(gt_point_classes),
            gt_trajectories=gt_result,
        )

        return item
