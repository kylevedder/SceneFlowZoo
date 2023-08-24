from scene_trajectory_benchmark.datastructures import PointCloudFrame, RGBFrame, RawSceneSequence, QuerySceneSequence, ResultsSceneSequence, SE3
from scene_trajectory_benchmark.datasets import *
from pathlib import Path
import torch
from typing import Tuple, Dict, List, Any
from pointclouds import to_fixed_array
import numpy as np
import time
from dataclasses import dataclass

@dataclass
class SceneTrajectoryBenchmarkSceneFlowItem():
    dataset_idx : int
    source_pc : torch.FloatTensor
    target_pc : torch.FloatTensor
    source_pose : SE3
    target_pose : SE3
    full_percept_pcs : List[torch.FloatTensor]
    full_percept_poses : List[SE3]
    query : QuerySceneSequence
    gt_flowed_source_pc : torch.FloatTensor
    gt_pc_class_mask : torch.LongTensor
    gt_result : ResultsSceneSequence

    def to(self, device):
        self.source_pc = self.source_pc.to(device)
        self.target_pc = self.target_pc.to(device)
        self.full_percept_pcs = [pc.to(device) for pc in self.full_percept_pcs]
        self.gt_flowed_source_pc = self.gt_flowed_source_pc.to(device)
        self.gt_pc_class_mask = self.gt_pc_class_mask.to(device)
        return self



class SceneTrajectoryBenchmarkSceneFlowDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset_name: str,
                 root_dir: Path):
        self.dataset = self._construct_dataset(dataset_name, dict(root_dir=root_dir))

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
            query.query_timestamps
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

            if timestamp in query.query_timestamps:
                problem_pc_arrays.append(pc_array)
                problem_poses.append(pose)

        assert len(full_percept_pc_arrays) == len(
            full_percept_poses
        ), f"Percept arrays and poses have different lengths."
        assert len(problem_pc_arrays) == len(
            problem_poses), f"Percept arrays and poses have different lengths."
        assert len(problem_pc_arrays) == len(
            query.query_timestamps
        ), f"Percept arrays and poses have different lengths."

        return problem_pc_arrays, problem_poses, full_percept_pc_arrays, full_percept_poses

    def _process_result(self, result: ResultsSceneSequence):
        flowed_source_pc = result.particle_trajectories.world_points[:, 1].astype(np.float32)
        point_cls_array = result.particle_trajectories.cls_ids
        return flowed_source_pc, point_cls_array

    def __getitem__(self, idx):
        dataset_entry: Tuple[QuerySceneSequence,
                             ResultsSceneSequence] = self.dataset[idx]
        query, gt_result = dataset_entry

        (source_pc, target_pc), (source_pose, target_pose), full_pc_points_list, full_pc_poses_list = self._process_query(
            query)

        gt_flowed_source_pc, gt_point_classes = self._process_result(gt_result)

        item = SceneTrajectoryBenchmarkSceneFlowItem(
            dataset_idx=idx,
            source_pc=torch.from_numpy(source_pc),
            target_pc=torch.from_numpy(target_pc),
            source_pose=source_pose,
            target_pose=target_pose,
            full_percept_pcs=[torch.from_numpy(pc) for pc in full_pc_points_list],
            full_percept_poses=full_pc_poses_list,
            query=query,
            gt_flowed_source_pc=torch.from_numpy(gt_flowed_source_pc),
            gt_pc_class_mask=torch.from_numpy(gt_point_classes),
            gt_result=gt_result
                    )

        return item

