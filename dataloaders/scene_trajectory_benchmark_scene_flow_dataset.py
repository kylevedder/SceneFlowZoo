from scene_trajectory_benchmark.datastructures import PointCloudFrame, RGBFrame, RawSceneSequence, QuerySceneSequence, SE3, GroundTruthParticleTrajectories, Timestamp
from scene_trajectory_benchmark.datasets import *
from pathlib import Path
import torch
from typing import Tuple, Dict, List, Any
from pointclouds import to_fixed_array, from_fixed_array
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class SceneTrajectoryBenchmarkSceneFlowItem():
    dataset_idx : int
    query_timestamp : Timestamp
    pc_array_stack : torch.FloatTensor
    pose_array_stack : torch.FloatTensor
    full_percept_pcs_array_stack : torch.FloatTensor
    full_percept_pose_array_stack : torch.FloatTensor
    gt_flowed_source_pc : torch.FloatTensor
    gt_pc_class_mask : torch.LongTensor
    gt_trajectories : GroundTruthParticleTrajectories

    def to(self, device):
        self.pc_array_stack = self.pc_array_stack.to(device)
        self.pose_array_stack = self.pose_array_stack.to(device)
        self.full_percept_pcs_array_stack = self.full_percept_pcs_array_stack.to(device)
        self.full_percept_pose_array_stack = self.full_percept_pose_array_stack.to(device)
        self.gt_flowed_source_pc = self.gt_flowed_source_pc.to(device)
        self.gt_pc_class_mask = self.gt_pc_class_mask.to(device)
        return self
    
    @property
    def source_pc(self):
        return from_fixed_array(self.pc_array_stack[:, 0])
    
    @property
    def target_pc(self):
        return from_fixed_array(self.pc_array_stack[:, 1])
    
    @property
    def source_pose(self) -> SE3:
        return SE3.from_array(self.pose_array_stack[:, 0])
    
    @property
    def target_pose(self) -> SE3:
        return SE3.from_array(self.pose_array_stack[:, 1])



class SceneTrajectoryBenchmarkSceneFlowDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_name: str,
                 root_dir: Path, 
                 max_pc_points: int = 120000, **kwargs):
        self.dataset = self._construct_dataset(dataset_name, dict(root_dir=root_dir, **kwargs))
        self.max_pc_points = max_pc_points

    def _construct_dataset(self, dataset_name: str, arguments : dict):
        dataset_name = dataset_name.lower()
        match dataset_name:
            case "argoverse2sceneflow":
                return Argoverse2SceneFlow(**arguments)
            
        raise ValueError(f"Unknown dataset name {dataset_name}")

    def __len__(self):
        return len(self.dataset)
    
    def evaluator(self):
        return self.dataset.evaluator()
    
    def collate_fn(self, batch : List[SceneTrajectoryBenchmarkSceneFlowItem]):
        return batch
        
    def _process_query(self, query: QuerySceneSequence):
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
        dataset_entry: Tuple[QuerySceneSequence,
                             GroundTruthParticleTrajectories] = self.dataset[idx]
        query, gt_result = dataset_entry

        (source_pc, target_pc), (source_pose, target_pose), full_pc_points_list, full_pc_poses_list = self._process_query(
            query)

        gt_flowed_source_pc, gt_point_classes = self._process_gt(gt_result)

        pc_array_stack = np.stack([to_fixed_array(source_pc, self.max_pc_points), 
                                   to_fixed_array(target_pc, self.max_pc_points)], axis=1)
        pose_array_stack = np.stack([source_pose.to_array(), target_pose.to_array()], axis=1)
        full_percept_pcs_array_stack = np.stack([to_fixed_array(pc, self.max_pc_points) for pc in full_pc_points_list], axis=1)
        full_percept_pose_array_stack = np.stack([pose.to_array() for pose in full_pc_poses_list], axis=1)

        item = SceneTrajectoryBenchmarkSceneFlowItem(
            dataset_idx=idx,
            query_timestamp=query.query_particles.query_init_timestamp,
            pc_array_stack=torch.from_numpy(pc_array_stack),
            pose_array_stack=torch.from_numpy(pose_array_stack),
            full_percept_pcs_array_stack=torch.from_numpy(full_percept_pcs_array_stack),
            full_percept_pose_array_stack=torch.from_numpy(full_percept_pose_array_stack),
            gt_flowed_source_pc=torch.from_numpy(gt_flowed_source_pc),
            gt_pc_class_mask=torch.from_numpy(gt_point_classes),
            gt_trajectories=gt_result
        )

        return item

