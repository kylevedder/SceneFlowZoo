import torch
import torch.nn as nn
from dataloaders import (
    TorchFullFrameInputSequence,
    TorchFullFrameOutputSequence,
    RawFullFrameInputSequence,
    RawFullFrameOutputSequence,
)
from abc import ABC, abstractmethod
from pointclouds import transform_pc
from pytorch_lightning.loggers import Logger
import enum
from models.components.optimization.cost_functions import BaseCostProblem
from typing import Union


class ForwardMode(enum.Enum):
    TRAIN = "train"
    VAL = "val"


class BaseRawModel(ABC, nn.Module):

    def forward(
        self,
        forward_mode: ForwardMode,
        batched_sequence: list[RawFullFrameInputSequence],
        logger: Logger,
    ) -> list[RawFullFrameOutputSequence]:
        return self.inference_forward(batched_sequence, logger)

    def inference_forward(
        self,
        batched_sequence: list[RawFullFrameInputSequence],
        logger: Logger,
    ) -> list[RawFullFrameOutputSequence]:
        return [
            self.inference_forward_single(input_sequence, logger)
            for input_sequence in batched_sequence
        ]

    @abstractmethod
    def inference_forward_single(
        self,
        input_sequence: RawFullFrameInputSequence,
        logger: Logger,
    ) -> RawFullFrameOutputSequence:
        raise NotImplementedError()

    @abstractmethod
    def loss_fn(
        self,
        input_batch: list[RawFullFrameInputSequence],
        model_res: list[RawFullFrameOutputSequence],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError()


class BaseTorchModel(ABC, nn.Module):
    def forward(
        self,
        forward_mode: ForwardMode,
        batched_sequence: list[TorchFullFrameInputSequence],
        logger: Logger,
    ) -> list[TorchFullFrameOutputSequence]:
        return self.inference_forward(batched_sequence, logger)

    def inference_forward(
        self,
        batched_sequence: list[TorchFullFrameInputSequence],
        logger: Logger,
    ) -> list[TorchFullFrameOutputSequence]:
        return [
            self.inference_forward_single(input_sequence, logger)
            for input_sequence in batched_sequence
        ]

    def inference_forward_single(
        self,
        input_sequence: TorchFullFrameInputSequence,
        logger: Logger,
    ) -> TorchFullFrameOutputSequence:
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

        assert (
            global_pc.shape == global_flow.shape
        ), f"Expected same shape, got {global_pc.shape} != {global_flow.shape}"

        flowed_global_pc = global_pc + global_flow

        # Transform both the point cloud and the flowed point cloud into the ego frame
        ego_pc = transform_pc(global_pc, torch.inverse(ego_to_global))
        ego_flowed_pc = transform_pc(flowed_global_pc, torch.inverse(ego_to_global))
        ego_flow = ego_flowed_pc - ego_pc
        return ego_flow

    @abstractmethod
    def loss_fn(
        self,
        input_batch: list[TorchFullFrameInputSequence],
        model_res: list[TorchFullFrameOutputSequence],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError()


class BaseOptimizationModel(BaseTorchModel):

    def __init__(self, full_input_sequence: TorchFullFrameInputSequence) -> None:
        super().__init__()
        self.full_input_sequence = full_input_sequence

    def forward(
        self,
        forward_mode: ForwardMode,
        batched_sequence: list[TorchFullFrameInputSequence],
        logger: Logger,
    ) -> list[BaseCostProblem] | list[TorchFullFrameOutputSequence]:
        match forward_mode:
            case ForwardMode.TRAIN:
                return self.optim_forward(batched_sequence, logger)
            case ForwardMode.VAL:
                return self.inference_forward(batched_sequence, logger)

    def optim_forward(
        self,
        batched_sequence: list[TorchFullFrameInputSequence],
        logger: Logger,
    ) -> list[BaseCostProblem]:
        return [
            self.optim_forward_single(input_sequence, logger) for input_sequence in batched_sequence
        ]

    @abstractmethod
    def optim_forward_single(
        self,
        input_sequence: TorchFullFrameInputSequence,
        logger: Logger,
    ) -> BaseCostProblem:
        raise NotImplementedError()

    def loss_fn(
        self,
        input_batch: list[TorchFullFrameInputSequence],
        model_res: list[BaseCostProblem],
    ) -> dict[str, torch.Tensor]:
        assert len(input_batch) == len(
            model_res
        ), f"Expected same length, got {len(input_batch)} != {len(model_res)}"
        assert len(model_res) > 0, f"Expected non-empty list, got {len(model_res)}"

        # Ensure that all the entries in model_res are actually subtypes of BaseCostProblem
        for e in model_res:
            assert isinstance(e, BaseCostProblem), f"Expected BaseCostProblem, got {type(e)}"

        loss = sum(e.cost() for e in model_res)
        return {
            "loss": loss,
        }


class AbstractBatcher(ABC):

    def __init__(self, full_sequence: TorchFullFrameInputSequence):
        self.full_sequence = full_sequence

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> TorchFullFrameInputSequence:
        raise NotImplementedError

    @abstractmethod
    def shuffle_minibatches(self, seed: int = 0):
        raise NotImplementedError

    # Implement iter method to allow for iteration over the batcher
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
