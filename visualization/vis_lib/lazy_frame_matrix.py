from bucketed_scene_flow_eval.interfaces import AbstractAVLidarSequence
from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame, TimeSyncedAVLidarData

from abc import ABC, abstractmethod


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class AbstractFrameMatrix(ABC):

    def __init__(self, sequences: list[TimeSyncedSceneFlowFrame], subsequence_length: int = 2):
        assert len(sequences) > 0, "At least one sequence must be provided."

        sequence_lengths = [len(sequence) for sequence in sequences]
        if not all(length == sequence_lengths[0] for length in sequence_lengths):
            print(
                bcolors.WARNING
                + f"Warning: Not all sequences have the same length. Lengths: {sequence_lengths}"
                + bcolors.ENDC
            )

        self.sequences = sequences
        self.subsequence_length = subsequence_length

    @property
    @abstractmethod
    def shape(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx_tuple: tuple[int, int]) -> list[TimeSyncedSceneFlowFrame]:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.shape[0]


class EagerFrameMatrix(AbstractFrameMatrix):

    @property
    def shape(self):
        return len(self.sequences), len(self.sequences[0])

    def __getitem__(self, idx_tuple: tuple[int, int]) -> list[TimeSyncedSceneFlowFrame]:
        sequence_idx, frame_idx = idx_tuple
        assert sequence_idx < len(
            self.sequences
        ), f"Invalid sequence index {sequence_idx} for len {len(self.sequences)}."
        sequence = self.sequences[sequence_idx]
        assert frame_idx < len(
            sequence
        ), f"Invalid frame index {frame_idx} for sequence of len {len(sequence)}."
        return sequence[frame_idx : frame_idx + self.subsequence_length]
