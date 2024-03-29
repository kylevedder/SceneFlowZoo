import torch
import torch.nn as nn
from dataloaders import BucketedSceneFlowInputSequence, BucketedSceneFlowOutputSequence
from abc import ABC, abstractmethod
from pointclouds import transform_pc


class BaseModel(ABC, nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, batched_sequence: list[BucketedSceneFlowInputSequence]
    ) -> list[BucketedSceneFlowOutputSequence]:
        raise NotImplementedError()

    def global_to_ego_flow(
        self, global_pc: torch.Tensor, global_flow: torch.Tensor, ego_to_global: torch.Tensor
    ) -> torch.Tensor:
        # Expect an Nx3 point cloud, an Nx3 global flow, and an 4x4 transformation matrix
        # Return an Nx3 ego flow

        assert global_pc.shape[1] == 3, f"Expected Nx3 point cloud, got {global_pc.shape}"
        assert global_flow.shape[1] == 3, f"Expected Nx3 global flow, got {global_flow.shape}"
        assert ego_to_global.shape == (
            4,
            4,
        ), f"Expected 4x4 transformation matrix, got {ego_to_global.shape}"

        flowed_global_pc = global_pc + global_flow

        # Transform both the point cloud and the flowed point cloud into the ego frame
        ego_pc = transform_pc(global_pc, torch.inverse(ego_to_global))
        ego_flowed_pc = transform_pc(flowed_global_pc, torch.inverse(ego_to_global))
        ego_flow = ego_flowed_pc - ego_pc
        return ego_flow
