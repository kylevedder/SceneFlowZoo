import torch
from dataloaders import TorchFullFrameInputSequence, TorchFullFrameOutputSequence
from bucketed_scene_flow_eval.interfaces import LoaderType
from pointclouds import to_fixed_array_torch
from .base_models import BaseTorchModel
from pytorch_lightning.loggers import Logger


class ConstantVectorBaseline(BaseTorchModel):

    def __init__(self, default_vector: torch.Tensor = torch.zeros(3)) -> None:
        super().__init__()
        # Convert numpy array to tensor
        if not isinstance(default_vector, torch.Tensor):
            default_vector = torch.tensor(default_vector, dtype=torch.float32)
        self.default_vector = default_vector
        assert self.default_vector.shape == (3,), "The default vector must be a 3D vector."

    def _single_frame_jagged_valid_pc_mask_to_jagged_flow(
        self,
        frame_full_pc_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert frame_full_pc_mask.ndim == 1, f"Expected 1D tensor, got {frame_full_pc_mask.ndim}."
        assert (
            frame_full_pc_mask.dtype == torch.bool
        ), f"Expected boolean tensor, got {frame_full_pc_mask.dtype}."
        N = frame_full_pc_mask.shape[0]
        flow_vector_buffer = (
            self.default_vector.expand((N, 3)).contiguous().to(frame_full_pc_mask.device)
        )
        flow_vector_buffer[~frame_full_pc_mask] = 0
        return flow_vector_buffer

    def _make_non_causal_output(
        self, input: TorchFullFrameInputSequence
    ) -> TorchFullFrameOutputSequence:
        K, PadN, _ = input.full_pc.shape
        # Default vector stacked to the same shape as the input point cloud, for all K-1 frames
        ego_flows = self.default_vector.expand((K - 1, PadN, 3))
        valid_flow = input.full_pc_mask[:-1]

        return TorchFullFrameOutputSequence(
            ego_flows=ego_flows,
            valid_flow_mask=valid_flow,
        )

    def _make_causal_output(
        self, input: TorchFullFrameInputSequence
    ) -> TorchFullFrameOutputSequence:
        K, PadN, _ = input.full_pc.shape
        jagged_full_pc_mask = input.get_full_pc_mask(-2)
        jagged_flow = self._single_frame_jagged_valid_pc_mask_to_jagged_flow(jagged_full_pc_mask)

        dense_flow = to_fixed_array_torch(jagged_flow, PadN)
        dense_valid_flow = to_fixed_array_torch(jagged_full_pc_mask, PadN)

        ego_flows = torch.unsqueeze(dense_flow, dim=0)
        valid_flow = torch.unsqueeze(dense_valid_flow, dim=0)
        return TorchFullFrameOutputSequence(
            ego_flows=ego_flows,
            valid_flow_mask=valid_flow,
        )

    def inference_forward_single(
        self, input: TorchFullFrameInputSequence, logger: Logger
    ) -> TorchFullFrameOutputSequence:

        if input.loader_type == LoaderType.NON_CAUSAL:
            return self._make_non_causal_output(input)
        elif input.loader_type == LoaderType.CAUSAL:
            return self._make_causal_output(input)
        else:
            raise ValueError(f"Unknown loader type: {input.loader_type}")

    def loss_fn(
        self,
        input_batch: list[TorchFullFrameInputSequence],
        model_res: list[TorchFullFrameOutputSequence],
    ) -> dict[str, torch.Tensor]:
        return {}
