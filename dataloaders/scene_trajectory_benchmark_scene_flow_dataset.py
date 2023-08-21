from scene_trajectory_benchmark.datastructures import PointCloudFrame, RGBFrame, RawSceneSequence, QuerySceneSequence, ResultsSceneSequence
from scene_trajectory_benchmark.datasets import *
from pathlib import Path
import torch
from typing import Tuple, Dict, List, Any
from pointclouds import to_fixed_array
import numpy as np
import time


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
        

        source_global_pc = query.scene_sequence[source_timestap][0].global_pc

        # assert np.allclose(source_global_pc, result.particle_trajectories.world_points[:, 0]), f"Source pointcloud in query {source_global_pc.shape} does not match source pointcloud in result {result.particle_trajectories.world_points[:, 0].shape}"

        flowed_source_pc = result.particle_trajectories.world_points[:, 0]

        # Build array, initialized to nan, for point classes using the length of the global PC
        point_cls_array = result.particle_trajectories.cls_ids.astype(np.float32)

        flowed_pc_array = to_fixed_array(flowed_source_pc.astype(np.float32),
                                            self.max_pc_points)
        flowed_pc_array_stack = np.stack([flowed_pc_array, flowed_pc_array], axis=0)
        point_cls_array = to_fixed_array(point_cls_array, self.max_pc_points)
        point_cls_array_stack = np.stack([point_cls_array, point_cls_array], axis=0)

        return flowed_pc_array_stack, point_cls_array_stack

    def __getitem__(self, idx, verbose : bool=False):

        if verbose:
            print(f"SceneTrajectoryBenchmarkSceneFlowDataset.__getitem__({idx}) start")
        dataset_entry: Tuple[QuerySceneSequence,
                             ResultsSceneSequence] = self.dataset[idx]
        query, gt_result = dataset_entry

        process_query_start = time.time()
        pc_array_stack, pose_array_stack, full_percept_pc_array_stack, full_percept_pose_array_stack = self._process_query(
            query)

        process_query_end = time.time()
        flowed_pc_array_stack, point_cls_array = self._process_result(query, gt_result)
        process_result_end = time.time()

        assert pc_array_stack.shape == flowed_pc_array_stack.shape, f"Pointcloud stacks have different shapes. {pc_array_stack.shape} != {flowed_pc_array_stack.shape}"
        if verbose:
            print(f"SceneTrajectoryBenchmarkSceneFlowDataset.__getitem__({idx}) process_query took {process_query_end - process_query_start} seconds")
            print(f"SceneTrajectoryBenchmarkSceneFlowDataset.__getitem__({idx}) process_result took {process_result_end - process_query_end} seconds")
        
            print(f"SceneTrajectoryBenchmarkSceneFlowDataset.__getitem__({idx}) end")

        return {
            "pc_array_stack": pc_array_stack,
            "pose_array_stack": pose_array_stack,
            "full_percept_pc_array_stack": full_percept_pc_array_stack,
            "full_percept_pose_array_stack": full_percept_pose_array_stack,
            "flowed_pc_array_stack": flowed_pc_array_stack,
            "pc_class_mask_stack": point_cls_array
        }
