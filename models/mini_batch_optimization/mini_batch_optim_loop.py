from models.whole_batch_optimization import WholeBatchOptimizationLoop
from models import AbstractBatcher
from dataloaders import TorchFullFrameInputSequence
import numpy as np
from dataclasses import dataclass
from bucketed_scene_flow_eval.interfaces import LoaderType


@dataclass
class MinibatchedSceneFlowInputSequence(TorchFullFrameInputSequence):
    full_sequence: TorchFullFrameInputSequence
    minibatch_idx: int


class SequenceMinibatcher(AbstractBatcher):
    def __init__(self, full_sequence: TorchFullFrameInputSequence, minibatch_size: int):
        assert (
            full_sequence.loader_type == LoaderType.NON_CAUSAL
        ), f"SequenceMinibatcher only supports non-causal datasets; constructed dataset has loader type {full_sequence.loader_type}."
        self.full_sequence = full_sequence
        self.minibatch_size = minibatch_size
        self.minibatch_order = self.shuffle_minibatches()

    def shuffle_minibatches(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self.minibatch_order = rng.permutation(len(self))
        return self.minibatch_order

    def __len__(self):
        # Length is the number of unique sliding windows of size minibatch_size we can extract.
        return len(self.full_sequence) - self.minibatch_size + 1

    def _get_item_true_idx(self, idx: int) -> MinibatchedSceneFlowInputSequence:
        slice_start_idx = idx
        slice_end_idx = idx + self.minibatch_size
        sliced = self.full_sequence.slice(slice_start_idx, slice_end_idx)
        return MinibatchedSceneFlowInputSequence(
            **vars(sliced), full_sequence=self.full_sequence, minibatch_idx=idx
        )

    def __getitem__(self, idx: int) -> MinibatchedSceneFlowInputSequence:
        return self._get_item_true_idx(self.minibatch_order[idx])


class MiniBatchOptimizationLoop(WholeBatchOptimizationLoop):

    def __init__(self, minibatch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.minibatch_size = minibatch_size

    def _setup_batcher(self, full_sequence: TorchFullFrameInputSequence) -> AbstractBatcher:
        return SequenceMinibatcher(full_sequence, self.minibatch_size)
