from scene_trajectory_benchmark.datastructures import PointCloudFrame, RGBFrame, RawSceneSequence, QuerySceneSequence, ResultsSceneSequence
from scene_trajectory_benchmark.datasets import *
from pathlib import Path
import torch
from typing import Tuple, Dict, List, Any
from pointclouds import to_fixed_array
import numpy as np


class SceneTrajectoryBenchmarkSceneFlowDataset(torch.utils.data.Dataset):

    def _construct_dataset(self, dataset_name: str, arguments : dict):
        dataset_name = dataset_name.lower()
        match dataset_name:
            case "argoverse2sceneflow":
                return Argoverse2SceneFlow(**arguments)
            
        raise ValueError(f"Unknown dataset name {dataset_name}")
            
        

    def __init__(self,
                 dataset_name: str,
                 root_dir: Path,
                 max_pc_points: int = 120000):
        self.dataset = self._construct_dataset(dataset_name, dict(root_dir=root_dir))
        self.max_pc_points = max_pc_points

    def __len__(self):
        return len(self.dataset)

    def _process_query(self, query: QuerySceneSequence):
        assert len(
            query.query_timestamps
        ) == 2, f"Query {query} has more than two timestamps. Only Scene Flow problems are supported."
        scene = query.scene_sequence

        # These contain the full problem percepts, not just the ones in the query.
        full_percept_pc_arrays: List[np.ndarray] = []
        full_percept_poses: List[np.ndarray] = []
        # These contain only the percepts in the query.0
        pc_arrays: List[np.ndarray] = []
        poses: List[np.ndarray] = []

        for timestamp in scene.get_percept_timesteps():
            pc_frame, rgb_frame = scene[timestamp]
            pc_array = to_fixed_array(pc_frame.global_pc.points,
                                      self.max_pc_points)
            pose_array = pc_frame.global_pose.to_array()

            full_percept_pc_arrays.append(pc_array)
            full_percept_poses.append(pose_array)

            if timestamp in query.query_timestamps:
                pc_arrays.append(pc_array)
                poses.append(pose_array)

        assert len(full_percept_pc_arrays) == len(
            full_percept_poses
        ), f"Percept arrays and poses have different lengths."
        assert len(pc_arrays) == len(
            poses), f"Percept arrays and poses have different lengths."
        assert len(pc_arrays) == len(
            query.query_timestamps
        ), f"Percept arrays and poses have different lengths."

        full_percept_pc_array_stack = np.stack(full_percept_pc_arrays, axis=0).astype(np.float32)
        full_percept_pose_array_stack = np.stack(full_percept_poses, axis=0).astype(np.float32)

        pc_array_stack = np.stack(pc_arrays, axis=0).astype(np.float32)
        pose_array_stack = np.stack(poses, axis=0).astype(np.float32)
        return pc_array_stack, pose_array_stack, full_percept_pc_array_stack, full_percept_pose_array_stack

    def _process_result(self, query: QuerySceneSequence,
                        result: ResultsSceneSequence):
        # Result contains trajectory information dictionary.
        # It's keyed by the index into the pointcloud, so we
        # can technically cheat and use that to index into the
        # pointcloud to update its flows.

        assert len(query.query_timestamps) == 2, f"Query {query} has more than two timestamps (has {len(query.query_timestamps)}). Only Scene Flow problems are supported."

        source_timestap = query.query_timestamps[0]
        # This is the key to the trajectory map.
        target_timestamp = query.query_timestamps[1]

        particle_ids, particle_trajectories = zip(
            *result.particle_trajectories.items())

        # These are the indexes to the source point cloud
        flowed_point_indices = np.array(particle_ids)

        # These are the new global frame locations of the points.
        flowed_point_locations = np.stack(
            [e.trajectory[target_timestamp].point for e in particle_trajectories],
            axis=0)
        # These are the classes of the points.
        point_classes = np.array(
            [1 if e.cls is not None else -1 for e in particle_trajectories], dtype=np.float32)
        
        

        source_pc_frame, _ = query.scene_sequence[source_timestap]

        # Build array for updated points using the standard global PC
        flowed_source_pc = source_pc_frame.global_pc.points.copy()
        flowed_source_pc[flowed_point_indices] = flowed_point_locations

        # Build array, initialized to false, for indicated flow points using the length of the global PC
        is_flowed = np.zeros(len(flowed_source_pc), dtype=bool)
        is_flowed[flowed_point_indices] = True

        # Build array, initialized to nan, for point classes using the length of the global PC
        point_cls_array = np.zeros(len(flowed_source_pc), dtype=np.float32) * -2
        point_cls_array[flowed_point_indices] = point_classes

        flowed_pc_array = to_fixed_array(flowed_source_pc.astype(np.float32),
                                            self.max_pc_points)
        flowed_pc_array_stack = np.stack([flowed_pc_array, flowed_pc_array], axis=0)
        is_flowed_array = to_fixed_array(is_flowed, self.max_pc_points)
        is_flowed_array_stack = np.stack([is_flowed_array, is_flowed_array], axis=0)
        point_cls_array = to_fixed_array(point_cls_array, self.max_pc_points)
        point_cls_array_stack = np.stack([point_cls_array, point_cls_array], axis=0)

        return flowed_pc_array_stack, is_flowed_array_stack, point_cls_array_stack

    def __getitem__(self, idx):
        dataset_entry: Tuple[QuerySceneSequence,
                             ResultsSceneSequence] = self.dataset[idx]
        query, gt_result = dataset_entry

        pc_array_stack, pose_array_stack, full_percept_pc_array_stack, full_percept_pose_array_stack = self._process_query(
            query)

        flowed_pc_array_stack, is_flowed, point_cls_array = self._process_result(query, gt_result)

        assert pc_array_stack.shape == flowed_pc_array_stack.shape, f"Pointcloud stacks have different shapes. {pc_array_stack.shape} != {flowed_pc_array_stack.shape}"

        return {
            "pc_array_stack": pc_array_stack,
            "pose_array_stack": pose_array_stack,
            "full_percept_pc_array_stack": full_percept_pc_array_stack,
            "full_percept_pose_array_stack": full_percept_pose_array_stack,
            "flowed_pc_array_stack": flowed_pc_array_stack,
            "is_flowed": is_flowed,
            "pc_class_mask_stack": point_cls_array
        }
