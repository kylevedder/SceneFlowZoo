import torch
import torch.nn as nn

import numpy as np
from models.embedders import DynamicVoxelizer
from models.nsfp_baseline import NSFPProcessor

from typing import Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import time


class NSFP(nn.Module):
    """
    FastFlow3D based on the paper:
    https://arxiv.org/abs/2103.01306v5

    Note that there are several small differences between this implementation and the paper:
     - We use a different loss function (predict flow for P_-1 to P_0 instead of P_0 to and 
       unseen P_1); referred to as pc0 and pc1 in the code.
    """

    def __init__(self,
                 VOXEL_SIZE,
                 POINT_CLOUD_RANGE,
                 SEQUENCE_LENGTH,
                 flow_save_folder: Path,
                 skip_existing: bool = False) -> None:
        super().__init__()
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        assert self.SEQUENCE_LENGTH == 2, "This implementation only supports a sequence length of 2."
        self.voxelizer = DynamicVoxelizer(voxel_size=VOXEL_SIZE,
                                          point_cloud_range=POINT_CLOUD_RANGE)
        self.nsfp_processor = NSFPProcessor()
        self.flow_save_folder = Path(flow_save_folder)
        self.flow_save_folder.mkdir(parents=True, exist_ok=True)
        self.skip_existing_cache = dict()
        self.skip_existing = skip_existing
        if self.skip_existing:
            self.skip_existing_cache = self._build_skip_existing_cache()

    def _build_skip_existing_cache(self):
        skip_existing_cache = dict()
        for log_id in self.flow_save_folder.iterdir():
            skip_existing_cache[log_id.name] = set()
            for npz_idx, npz_file in enumerate(sorted((log_id).glob("*.npz"))):
                npz_filename_idx = int(npz_file.stem.split('_')[0])
                skip_existing_cache[log_id.name].add(npz_idx)
                skip_existing_cache[log_id.name].add(npz_filename_idx)
        return skip_existing_cache

    def _save_result(self, log_id: str, batch_idx: int, minibatch_idx: int,
                     delta_time: float, flow: torch.Tensor,
                     valid_idxes: torch.Tensor):
        flow = flow.cpu().numpy()
        valid_idxes = valid_idxes.cpu().numpy()
        data = {
            'delta_time': delta_time,
            'flow': flow,
            'valid_idxes': valid_idxes,
        }
        seq_save_folder = self.flow_save_folder / log_id
        seq_save_folder.mkdir(parents=True, exist_ok=True)
        np.savez(seq_save_folder / f'{batch_idx:010d}_{minibatch_idx:03d}.npz',
                 **data)

    def _voxelize_batched_sequence(self, batched_sequence: Dict[str,
                                                                torch.Tensor]):
        pc_arrays = batched_sequence['pc_array_stack']
        pc0s = pc_arrays[:, 0]
        pc1s = pc_arrays[:, 1]

        pc0_voxel_infos_lst = self.voxelizer(pc0s)
        pc1_voxel_infos_lst = self.voxelizer(pc1s)

        pc0_points_lst = [e["points"] for e in pc0_voxel_infos_lst]
        pc0_valid_point_idxes = [e["point_idxes"] for e in pc0_voxel_infos_lst]
        pc1_points_lst = [e["points"] for e in pc1_voxel_infos_lst]
        pc1_valid_point_idxes = [e["point_idxes"] for e in pc1_voxel_infos_lst]

        return pc0_points_lst, pc0_valid_point_idxes, pc1_points_lst, pc1_valid_point_idxes

    def _visualize_result(self, pc0_points: torch.Tensor,
                          warped_pc0_points: torch.Tensor):

        # if pc0_points is torch tensor, convert to numpy
        if isinstance(pc0_points, torch.Tensor):
            pc0_points = pc0_points.cpu().numpy()[0]
        if isinstance(warped_pc0_points, torch.Tensor):
            warped_pc0_points = warped_pc0_points.cpu().numpy()[0]

        import open3d as o3d

        line_set = o3d.geometry.LineSet()
        assert len(pc0_points) == len(
            warped_pc0_points
        ), f'pc and flowed_pc must have same length, but got {len(pc0_pcd)} and {len(warped_pc0_points)}'
        line_set_points = np.concatenate([pc0_points, warped_pc0_points],
                                         axis=0)

        pc0_pcd = o3d.geometry.PointCloud()
        pc0_pcd.points = o3d.utility.Vector3dVector(pc0_points)
        warped_pc0_pcd = o3d.geometry.PointCloud()
        warped_pc0_pcd.points = o3d.utility.Vector3dVector(warped_pc0_points)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_set_points)
        lines = np.array([[i, i + len(pc0_points)]
                          for i in range(len(pc0_points))])
        line_set.lines = o3d.utility.Vector2iVector(lines)

        o3d.visualization.draw_geometries([pc0_pcd, warped_pc0_pcd, line_set])

    def forward(self, batched_sequence: Dict[str, torch.Tensor]):
        pc0_points_lst, pc0_valid_point_idxes, pc1_points_lst, pc1_valid_point_idxes = self._voxelize_batched_sequence(
            batched_sequence)

        # process minibatch
        flows = []
        delta_times = []
        for minibatch_idx, (pc0_points, pc1_points,
                            pc0_valid_point_idx) in enumerate(
                                zip(pc0_points_lst, pc1_points_lst,
                                    pc0_valid_point_idxes)):
            log_id = batched_sequence['log_ids'][minibatch_idx][0]
            batch_idx = batched_sequence['data_index'].item()
            if self.skip_existing:
                if log_id in self.skip_existing_cache:
                    existing_set = self.skip_existing_cache[log_id]
                    if batch_idx in existing_set:
                        print(f'Skip existing flow for {log_id} {batch_idx}')
                        continue
                    else:
                        print("batch_idx", batch_idx, "not in", existing_set)
                else:
                    print("log_id", log_id, "not in", self.skip_existing_cache.keys())
            #     and (
            #         log_id in self.skip_existing_cache) and (
            #             batch_idx in self.skip_existing_cache[log_id]):
            #     print(f'Skip existing flow for {log_id} {batch_idx}')
            #     continue
            # else:
            #     print(self.skip_existing_cache.keys())

            pc0_points = torch.unsqueeze(pc0_points, 0)
            pc1_points = torch.unsqueeze(pc1_points, 0)

            with torch.inference_mode(False):
                with torch.enable_grad():
                    pc0_points_new = pc0_points.clone().detach(
                    ).requires_grad_(True)
                    pc1_points_new = pc1_points.clone().detach(
                    ).requires_grad_(True)

                    self.nsfp_processor.train()

                    before_time = time.time()
                    warped_pc0_points, _ = self.nsfp_processor(
                        pc0_points_new, pc1_points_new, pc1_points_new.device)
                    after_time = time.time()

            delta_time = after_time - before_time
            # self._visualize_result(pc0_points, warped_pc0_points)
            flow = warped_pc0_points - pc0_points
            self._save_result(log_id=log_id,
                              batch_idx=batch_idx,
                              minibatch_idx=minibatch_idx,
                              delta_time=delta_time,
                              flow=flow,
                              valid_idxes=pc0_valid_point_idx)
            flows.append(flow.squeeze(0))
            delta_times.append(delta_time)

        return {
            "forward": {
                "flow": flows,
                "batch_delta_time": np.sum(delta_times),
                "pc0_points_lst": pc0_points_lst,
                "pc0_valid_point_idxes": pc0_valid_point_idxes,
                "pc1_points_lst": pc1_points_lst,
                "pc1_valid_point_idxes": pc1_valid_point_idxes
            }
        }


class NSFPCached(NSFP):

    def __init__(self, VOXEL_SIZE, POINT_CLOUD_RANGE, SEQUENCE_LENGTH,
                 flow_save_folder: Path) -> None:
        super().__init__(VOXEL_SIZE, POINT_CLOUD_RANGE, SEQUENCE_LENGTH,
                         flow_save_folder)
        # Implement basic caching to avoid repeated folder reads for the same log.
        self.flow_folder_id = None
        self.flow_folder_list = None

    def _load_result(self, log_id: str, log_idx: int):
        if self.flow_folder_id != log_id:
            self.flow_folder_id = log_id
            flow_folder = self.flow_save_folder / log_id
            assert flow_folder.is_dir(), f"{flow_folder} does not exist"
            self.flow_folder_list = sorted(flow_folder.glob('*.npz'))

        assert log_idx < len(
            self.flow_folder_list), f"Log index {log_idx} is out of range"
        flow_file = self.flow_folder_list[log_idx]
        data = dict(np.load(flow_file, allow_pickle=True))
        flow = data['flow']
        valid_idxes = data['valid_idxes']
        delta_time = data['delta_time']
        return flow, valid_idxes, delta_time

    def _transpose_list(self, lst):
        if isinstance(lst[0], torch.Tensor):
            return torch.stack(lst).T.tolist()
        return np.array(lst).T.tolist()

    def forward(self, batched_sequence: Dict[str, torch.Tensor]):
        pc0_points_lst, pc0_valid_point_idxes, pc1_points_lst, pc1_valid_point_idxes = self._voxelize_batched_sequence(
            batched_sequence)

        log_ids_lst = self._transpose_list(batched_sequence['log_ids'])
        log_idxes_lst = self._transpose_list(batched_sequence['log_idxes'])

        flows = []
        delta_times = []
        for minibatch_idx, (pc0_points, pc0_valid_points_idx, log_ids,
                            log_idxes) in enumerate(
                                zip(pc0_points_lst, pc0_valid_point_idxes,
                                    log_ids_lst, log_idxes_lst)):
            assert len(
                log_ids) == 2, f"Expect 2 frames for NSFP, got {len(log_ids)}"
            assert len(
                log_idxes
            ) == 2, f"Expect 2 frames for NSFP, got {len(log_idxes)}"
            flow, flow_valid_idxes, delta_time = self._load_result(
                log_ids[0], log_idxes[0])
            assert flow_valid_idxes.shape == pc0_valid_points_idx.shape, f"{flow_valid_idxes.shape} != {pc0_valid_points_idx.shape}"
            idx_equality = flow_valid_idxes == pc0_valid_points_idx.detach(
            ).cpu().numpy()
            assert idx_equality.all(
            ), f"Points that disagree: {(~idx_equality).astype(int).sum()}"
            assert flow.shape[0] == 1, f"{flow.shape[0]} != 1"
            flow = flow.squeeze(0)
            assert flow.shape == pc0_points.shape, f"{flow.shape} != {pc0_points.shape}"
            flows.append(torch.from_numpy(flow).to(pc0_points.device))
            delta_times.append(delta_time)

        return {
            "forward": {
                "flow": flows,
                "batch_delta_time": np.sum(delta_times),
                "pc0_points_lst": pc0_points_lst,
                "pc0_valid_point_idxes": pc0_valid_point_idxes,
                "pc1_points_lst": pc1_points_lst,
                "pc1_valid_point_idxes": pc1_valid_point_idxes
            }
        }
