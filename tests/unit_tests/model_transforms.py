from models import BaseModel
import torch
import numpy as np
from bucketed_scene_flow_eval.datastructures import SE3
import pytest
from dataloaders import BucketedSceneFlowInputSequence
from bucketed_scene_flow_eval.datasets import Argoverse2CausalSceneFlow
from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame
from pathlib import Path

class DummyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, batched_sequence):
        raise NotImplementedError()
    

@pytest.fixture
def frame_list_tuple() -> tuple[list[TimeSyncedSceneFlowFrame], BucketedSceneFlowInputSequence]:
    sequence_length = 5
    dataset_idx = 0
    dataset = Argoverse2CausalSceneFlow(
        root_dir=Path("/tmp/argoverse2_small/val/"), subsequence_length=sequence_length
    )
    frame_list = dataset[dataset_idx]
    torch_sequence = BucketedSceneFlowInputSequence.from_frame_list(
        idx=dataset_idx, frame_list=frame_list, pc_max_len=120000, loader_type=dataset.loader_type()
    )

    return frame_list, torch_sequence


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
    ego_flow_torch = model.global_to_ego_flow(global_pc_torch, global_flow_torch, ego_to_global_se3_torch)

    assert torch.allclose(ego_flow_torch, torch.from_numpy(ego_flow_np).float(), atol=1e-6), f"Expected {ego_flow_np}, got {ego_flow_torch}"


