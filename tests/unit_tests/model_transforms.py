from models import BaseModel
import torch
import numpy as np
from bucketed_scene_flow_eval.datastructures import SE3
import pytest
from dataloaders import (
    BucketedSceneFlowDataset,
    BucketedSceneFlowInputSequence,
    BucketedSceneFlowOutputSequence,
)
from bucketed_scene_flow_eval.datasets import Argoverse2CausalSceneFlow
from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame
from pathlib import Path
from pointclouds import to_fixed_array_torch


class DummyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()

    def _cheat_use_gt_flow(
        self, input: BucketedSceneFlowInputSequence
    ) -> BucketedSceneFlowOutputSequence:

        full_ego_flows = []
        full_flow_masks = []

        for idx in range(len(input) - 1):
            global_pc = input.get_full_global_pc(idx)
            gt_global_flow = input.get_full_global_pc_gt_flowed(idx) - global_pc
            gt_ego_flow = input.get_full_ego_pc_gt_flowed(idx) - input.get_full_ego_pc(idx)

            ego_to_global_se3 = input.pc_poses_ego_to_global[idx]
            trans_ego_flow = self.global_to_ego_flow(global_pc, gt_global_flow, ego_to_global_se3)

            assert torch.allclose(
                gt_ego_flow, trans_ego_flow, atol=2e-5
            ), f"Expected tensors to be close, but max difference was {torch.max(torch.abs(gt_ego_flow - trans_ego_flow))}"

            valid_flow_mask = input.get_full_pc_gt_flow_mask(idx)

            full_ego_flows.append(to_fixed_array_torch(trans_ego_flow, max_len=120000))
            full_flow_masks.append(to_fixed_array_torch(valid_flow_mask, max_len=120000))

        full_ego_flows_tensor = torch.stack(full_ego_flows)
        full_flow_masks_tensor = torch.stack(full_flow_masks)

        return BucketedSceneFlowOutputSequence(
            ego_flows=full_ego_flows_tensor, valid_flow_mask=full_flow_masks_tensor
        )

    def forward(
        self, batched_sequence: list[BucketedSceneFlowInputSequence], logger
    ) -> list[BucketedSceneFlowOutputSequence]:
        return [self._cheat_use_gt_flow(input) for input in batched_sequence]

    def loss_fn(
        self,
        input_batch: list[BucketedSceneFlowInputSequence],
        model_res: list[BucketedSceneFlowOutputSequence],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError()


@pytest.fixture
def dataset_with_gt_flow() -> BucketedSceneFlowDataset:
    dataset = BucketedSceneFlowDataset(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=Path("/tmp/argoverse2_small/val/"),
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=5,
    )

    return dataset


def test_global_to_ego_flow_basic():

    # Single point PC at 1,2,3
    ego_pc_np = np.array([[1, 2, 3]])

    # Single flow vector at 0,1,0
    ego_flow_np = np.array([[0, 1, 0]])

    ego_to_global_se3 = SE3.from_rot_x_y_z_translation_x_y_z(
        rx=0, ry=0, rz=np.deg2rad(30), tx=0, ty=0, tz=0
    )

    global_flow_np = ego_to_global_se3.transform_flow(ego_flow_np)

    global_pc_np = ego_to_global_se3.transform_points(ego_pc_np)

    global_flow_torch = torch.from_numpy(global_flow_np).float()
    global_pc_torch = torch.from_numpy(global_pc_np).float()
    ego_to_global_se3_torch = torch.from_numpy(ego_to_global_se3.to_array()).float()

    model = DummyModel()
    ego_flow_torch = model.global_to_ego_flow(
        global_pc_torch, global_flow_torch, ego_to_global_se3_torch
    )

    assert torch.allclose(
        ego_flow_torch, torch.from_numpy(ego_flow_np).float(), atol=1e-6
    ), f"Expected {ego_flow_np}, got {ego_flow_torch}"


def test_validate_cheating_with_gt_flow_is_perfect(dataset_with_gt_flow: BucketedSceneFlowDataset):
    model = DummyModel()
    for idx in range(len(dataset_with_gt_flow)):
        input = dataset_with_gt_flow[idx]
        out_lst = model([input], None)
        output: BucketedSceneFlowOutputSequence = out_lst[0]

        for idx in range(len(input) - 1):
            gt_ego_flow = input.get_full_ego_pc_gt_flowed(idx) - input.get_full_ego_pc(idx)
            pred_ego_flow = output.get_full_ego_flow(idx)

            assert torch.allclose(
                pred_ego_flow, gt_ego_flow, atol=2e-5
            ), f"Expected tensors to be close, but max difference was {torch.max(torch.abs(pred_ego_flow - gt_ego_flow))}"
