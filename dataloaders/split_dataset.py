from dataclasses import dataclass

from bucketed_scene_flow_eval.interfaces import AbstractDataset
from .abstract_dataset import BaseDataset
from pathlib import Path
from bucketed_scene_flow_eval.interfaces import LoaderType


@dataclass
class FullFrameDatasetSplit:
    split_idx: int
    num_splits: int

    def __post_init__(self):
        # Type check the split index and number of splits.
        assert isinstance(self.split_idx, int), f"Invalid split index type {type(self.split_idx)}."
        assert isinstance(
            self.num_splits, int
        ), f"Invalid number of splits type {type(self.num_splits)}."
        assert (
            0 <= self.split_idx < self.num_splits
        ), f"Invalid split index {self.split_idx} for {self.num_splits} splits."

    def current_split_length(self, total_length: int) -> int:
        # All splits should be the same base length but
        # the remainder should be distributed across the splits.
        base_length = total_length // self.num_splits
        distributed_remainder = 1 if total_length % self.num_splits > self.split_idx else 0
        return base_length + distributed_remainder

    def _current_split_global_start(self, total_length: int) -> int:
        base_length = total_length // self.num_splits
        distributed_remainder = min(total_length % self.num_splits, self.split_idx)
        return base_length * self.split_idx + distributed_remainder

    def split_index_to_global_index(self, split_idx: int, global_length: int) -> int:
        assert (
            0 <= split_idx < self.current_split_length(global_length)
        ), f"Invalid split index {split_idx}."
        return self._current_split_global_start(global_length) + split_idx


class SplitFullFrameDataset(BaseDataset):

    def __init__(
        self,
        set_length: int | None = None,
        split: FullFrameDatasetSplit | dict[str, int] = FullFrameDatasetSplit(0, 1),
    ):
        self.set_length = set_length
        self.split = (
            split if isinstance(split, FullFrameDatasetSplit) else FullFrameDatasetSplit(**split)
        )
        self.dataset: AbstractDataset

    def _global_len(self):
        global_length = len(self.dataset)
        if self.set_length is not None:
            global_length = min(global_length, self.set_length)
        return global_length

    def __len__(self):
        return self.split.current_split_length(self._global_len())

    def _get_global_idx(self, split_idx):
        assert isinstance(split_idx, int), f"Index must be an integer. Got {type(split_idx)}."
        assert 0 <= split_idx < len(self), "Index out of range."
        return self.split.split_index_to_global_index(split_idx, self._global_len())

    def loader_type(self) -> LoaderType:
        return self.dataset.loader_type()
