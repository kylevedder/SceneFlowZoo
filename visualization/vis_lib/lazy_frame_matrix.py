from bucketed_scene_flow_eval.interfaces import AbstractAVLidarSequence
from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData

from abc import ABC, abstractmethod


class AbstractFrameMatrix(ABC):

    @abstractmethod
    def __init__(self, sequences: list[AbstractAVLidarSequence], subsequence_length: int = 2):
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(
        self, idx_tuple: tuple[int, int]
    ) -> list[tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.shape[0]


class CausalLazyFrameMatrix(AbstractFrameMatrix):
    def __init__(self, sequences: list[AbstractAVLidarSequence], subsequence_length: int = 2):
        assert len(sequences) > 0, "At least one sequence must be provided."
        # Ensure all the sequences have the same length
        for sequence in sequences[1:]:
            assert len(sequence) == len(
                sequences[0]
            ), f"All sequences must have the same length. Got {len(sequence)} and {len(sequences[0])}."
        self.sequences = sequences
        self.subsequence_length = subsequence_length

    @property
    def shape(self):
        return len(self.sequences), len(self.sequences[0])

    def __getitem__(
        self, idx_tuple: tuple[int, int]
    ) -> list[tuple[TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData]]:
        sequence_idx, frame_idx = idx_tuple
        assert sequence_idx < len(
            self.sequences
        ), f"Invalid sequence index {sequence_idx} for len {len(self.sequences)}."
        sequence = self.sequences[sequence_idx]

        def rel_idx_to_with_flow(relative_idx: int) -> bool:
            return relative_idx == (self.subsequence_length - 2)

        return [
            sequence.load(idx, relative_to_idx=0, with_flow=rel_idx_to_with_flow(idx - frame_idx))
            for idx in range(frame_idx, frame_idx + self.subsequence_length)
        ]
