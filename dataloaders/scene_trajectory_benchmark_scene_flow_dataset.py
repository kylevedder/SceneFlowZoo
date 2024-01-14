from .dataclasses import BucketedSceneFlowItem
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from bucketed_scene_flow_eval.datasets import Argoverse2SceneFlow
from bucketed_scene_flow_eval.datastructures import (
    SE3,
    GroundTruthPointFlow,
    QuerySceneSequence,
    RawSceneSequence,
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

    def _extract_all_percept_arrays(
        self, scene: RawSceneSequence
    ) -> Tuple[List[np.ndarray], List[SE3]]:
        # These contain the full problem percepts, not just the ones in the query.
        all_percept_pc_arrays: List[np.ndarray] = []
        all_percept_poses: List[SE3] = []

        for timestamp in scene.get_percept_timesteps():
            pc_frame = scene[timestamp].pc_frame
            pc_array = pc_frame.global_pc.points.astype(np.float32)
            pose = pc_frame.global_pose

            all_percept_pc_arrays.append(pc_array)
            all_percept_poses.append(pose)

        return all_percept_pc_arrays, all_percept_poses

    def _process_query(
        self, query: QuerySceneSequence
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[SE3, SE3]]:
        assert (
            len(query.query_flow_timestamps) == 2
        ), f"Query {query} has more than two timestamps. Only Scene Flow problems are supported."
        scene = query.scene_sequence

        # These contain the percepts for the query only
        source_timestamp, target_timestamp = query.query_flow_timestamps

        # Source PC
        source_pc_frame = scene[source_timestamp].pc_frame
        # Grab the full, unmasked point cloud.
        source_pc_array = source_pc_frame.full_global_pc.points.astype(np.float32)
        source_pc_mask = source_pc_frame.mask
        source_pose = source_pc_frame.global_pose

        # Target PC
        target_pc_frame = scene[target_timestamp].pc_frame
        # Grab the already masked point cloud.
        target_pc_array = target_pc_frame.full_global_pc.points.astype(np.float32)
        target_pc_mask = target_pc_frame.mask
        target_pose = target_pc_frame.global_pose

        # These contain only the percepts in the query.
        query_pc_arrays = (source_pc_array, target_pc_array)
        query_pc_masks = (source_pc_mask, target_pc_mask)
        query_poses = (source_pose, target_pose)

        return query_pc_arrays, query_pc_masks, query_poses

    def _process_gt(self, result: GroundTruthPointFlow):
        flowed_source_pc = result.world_points[:, 1].astype(np.float32)
        is_valid_mask = result.is_valid_flow
        point_cls_array = result.cls_ids
        return flowed_source_pc, is_valid_mask, point_cls_array

    def __getitem__(self, idx):
        dataset_entry: Tuple[QuerySceneSequence, GroundTruthPointFlow] = self.dataset[idx]
        query, gt_result = dataset_entry

        all_pc_points_list, all_pc_poses_list = self._extract_all_percept_arrays(
            query.scene_sequence
        )

        all_percept_pcs_array_stack = np.stack(
            [to_fixed_array(pc, self.max_pc_points) for pc in all_pc_points_list], axis=0
        )
        all_percept_pose_array_stack = np.stack(
            [pose.to_array() for pose in all_pc_poses_list], axis=0
        )

        (
            (
                raw_source_pc,
                raw_target_pc,
            ),
            (
                raw_source_pc_mask,
                raw_target_pc_mask,
            ),
            (
                source_pose,
                target_pose,
            ),
        ) = self._process_query(query)

        gt_flowed_source_pc, gt_valid_flow_mask, gt_point_classes = self._process_gt(gt_result)

        item = BucketedSceneFlowItem(
            dataset_log_id=query.scene_sequence.log_id,
            dataset_idx=idx,
            query_timestamp=query.query_particles.query_init_timestamp,
            raw_source_pc=torch.from_numpy(raw_source_pc),
            raw_source_pc_mask=torch.from_numpy(raw_source_pc_mask),
            raw_target_pc=torch.from_numpy(raw_target_pc),
            raw_target_pc_mask=torch.from_numpy(raw_target_pc_mask),
            source_pose=source_pose,
            target_pose=target_pose,
            all_percept_pcs_array_stack=torch.from_numpy(all_percept_pcs_array_stack),
            all_percept_pose_array_stack=torch.from_numpy(all_percept_pose_array_stack),
            raw_gt_flowed_source_pc=torch.from_numpy(gt_flowed_source_pc),
            raw_gt_pc_class_mask=torch.from_numpy(gt_point_classes),
            raw_gt_flowed_source_pc_mask=torch.from_numpy(gt_valid_flow_mask),
            gt_trajectories=gt_result,
        )

        return item
