import torch
import torch.nn as nn

import numpy as np
from models.embedders import DynamicVoxelizer
from models.nsfp_baseline import NSFPProcessor
from pointclouds import PointCloud, warped_pc_loss, pc0_to_pc1_distance

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

    def __init__(self, VOXEL_SIZE, POINT_CLOUD_RANGE, SEQUENCE_LENGTH,
                 flow_save_folder: Path) -> None:
        super().__init__()
        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        assert self.SEQUENCE_LENGTH == 2, "This implementation only supports a sequence length of 2."
        self.voxelizer = DynamicVoxelizer(voxel_size=VOXEL_SIZE,
                                          point_cloud_range=POINT_CLOUD_RANGE)
        self.nsfp_processor = NSFPProcessor()
        self.flow_save_folder = Path(flow_save_folder)
        self.flow_save_folder.mkdir(parents=True, exist_ok=True)

    def _save_result(self, log_id: str, batch_idx: int, minibatch_idx: int,
                     delta_time: float, flow: torch.Tensor):
        flow = flow.cpu().numpy()
        data = {
            'delta_time': delta_time,
            'flow': flow,
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

    def forward(self, batched_sequence: Dict[str, torch.Tensor]):
        pc0_points_lst, pc0_valid_point_idxes, pc1_points_lst, pc1_valid_point_idxes = self._voxelize_batched_sequence(
            batched_sequence)

        # process minibatch
        flows = []
        for minibatch_idx, (pc0_points, pc1_points) in enumerate(
                zip(pc0_points_lst, pc1_points_lst)):
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
            flow = warped_pc0_points - pc0_points
            self._save_result(
                log_id=batched_sequence['log_ids'][minibatch_idx][0],
                batch_idx=batched_sequence['data_index'].item(),
                minibatch_idx=minibatch_idx,
                delta_time=delta_time,
                flow=flow)
            flows.append(flow.squeeze(0))

        return {
            "forward": {
                "flow": flows,
                "pc0_points_lst": pc0_points_lst,
                "pc0_valid_point_idxes": pc0_valid_point_idxes,
                "pc1_points_lst": pc1_points_lst,
                "pc1_valid_point_idxes": pc1_valid_point_idxes
            }
        }


class NSFPCached(NSFP):

    def _load_result(self, batch_idx: int, minibatch_idx: int):
        data = np.load(self.flow_save_folder /
                       f'{batch_idx:010d}_{minibatch_idx:03d}.npz',
                       allow_pickle=True)
        flow = data['flow']
        return flow

    def _load_low_prec_result(self, batch_idx: int, minibatch_idx: int):
        data = np.load(Path("/bigdata/nsfp_results_low_prec") /
                       f'{batch_idx:010d}_{minibatch_idx:03d}.npz',
                       allow_pickle=True)
        flow = data['flow']
        return flow

    def forward(self, batched_sequence: Dict[str, torch.Tensor]):
        pc0_points_lst, pc0_valid_point_idxes, pc1_points_lst, pc1_valid_point_idxes = self._voxelize_batched_sequence(
            batched_sequence)

        flows = []
        for minibatch_idx, (pc0_points, pc1_points) in enumerate(
                zip(pc0_points_lst, pc1_points_lst)):
            flow = self._load_result(
                batch_idx=batched_sequence['data_index'].item(),
                minibatch_idx=minibatch_idx)
            low_prec_flow = self._load_low_prec_result(
                batch_idx=batched_sequence['data_index'].item(),
                minibatch_idx=minibatch_idx)
            # flow = low_prec_flow
            flows.append(
                torch.from_numpy(flow.squeeze(0)).to(pc0_points.device))

        return {
            "flow": flows,
            "pc0_points_lst": pc0_points_lst,
            "pc0_valid_point_idxes": pc0_valid_point_idxes,
            "pc1_points_lst": pc1_points_lst,
            "pc1_valid_point_idxes": pc1_valid_point_idxes
        }
