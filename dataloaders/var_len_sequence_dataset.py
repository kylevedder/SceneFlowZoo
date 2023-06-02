import torch
import numpy as np
from tqdm import tqdm
import enum
from typing import Union, List, Tuple, Dict, Optional, Any
from dataloaders.sequence_dataset import OriginMode
from pathlib import Path
from .sequence_dataset import SubsequenceRawDataset, SubsequenceSupervisedFlowDataset, SubsequenceUnsupervisedFlowDataset
from functools import partial


class VarLenSubsequenceRawDataset(SubsequenceRawDataset):

    def __init__(self,
                 sequence_loader,
                 subsequence_length: int,
                 origin_mode: Union[OriginMode, str],
                 max_pc_points: int = 90000,
                 subset_fraction: float = 1.0,
                 shuffle=False):
        self.sequence_loader = sequence_loader

        # Subsequence length is the number of pointclouds projected into the given frame.
        assert subsequence_length > 0, f"subsequence_length must be > 0, got {subsequence_length}"
        self.subsequence_length = subsequence_length

        if isinstance(origin_mode, str):
            origin_mode = OriginMode[origin_mode]
        self.origin_mode = origin_mode
        self.max_pc_points = max_pc_points

        self.index_array_range, self.sequence_list = self._build_sequence_lookup(
        )
        self.shuffled_idx_lookup = self._build_shuffle_lookup(
            shuffle, subset_fraction)

    def _build_sequence_lookup(self):
        ids = sorted(self.sequence_loader.get_sequence_ids())
        index_range_list = [0]
        sequence_list = []
        for id in ids:
            sequence = self.sequence_loader.load_sequence(id)

            # Get number of unique subsequences in this sequence.
            sequence_length = len(sequence)
            num_subsequences = sequence_length - self.subsequence_length + 1

            index_range_list.append(index_range_list[-1] + num_subsequences)
            sequence_list.append(sequence)

        index_range_array = np.array(index_range_list)
        return index_range_array, sequence_list

    def _build_shuffle_lookup(self, shuffle, subset_fraction):
        shuffled_idx_lookup = np.arange(self.index_array_range[-1])
        if shuffle:
            np.random.shuffle(shuffled_idx_lookup)

        assert 1.0 >= subset_fraction > 0.0, f"subset_fraction must be in (0.0, 1.0], got {subset_fraction}"
        if subset_fraction == 1.0:
            return shuffled_idx_lookup
        max_index = int(len(shuffled_idx_lookup) * subset_fraction)
        print(
            f"Using only {max_index} of {len(shuffled_idx_lookup)} sequences.")
        return shuffled_idx_lookup[:max_index]

    def __len__(self):
        return len(self.shuffled_idx_lookup)

    def _global_idx_to_seq_and_seq_idx(self, input_global_idx):
        assert input_global_idx >= 0 and input_global_idx < len(
            self
        ), f"global_idx must be >= 0 and < len(self), got {input_global_idx} and {len(self)}"

        global_idx = self.shuffled_idx_lookup[input_global_idx]

        # Find the sequence that contains this index. self.index_array_range provides a
        # sorted global index range table, whose index can extract the relevant sequence
        # from self.sequence_list.
        seq_idx = np.searchsorted(
            self.index_array_range, global_idx, side='right') - 1
        assert seq_idx >= 0 and seq_idx < len(
            self.sequence_list
        ), f"seq_idx must be >= 0 and < len(self.sequence_list), got {seq_idx} and {len(self.sequence_list)}"

        sequence = self.sequence_list[seq_idx]
        sequence_idx = global_idx - self.index_array_range[seq_idx]

        assert sequence_idx >= 0 and sequence_idx < len(
            sequence
        ), f"sequence_idx must be >= 0 and < len(sequence), got {sequence_idx} and {len(sequence)}"
        return sequence, sequence_idx

    def _get_subsequence(self, global_idx):

        sequence, subsequence_begin_index = self._global_idx_to_seq_and_seq_idx(
            global_idx)

        if self.origin_mode == OriginMode.FIRST_ENTRY:
            origin_idx = subsequence_begin_index
        elif self.origin_mode == OriginMode.LAST_ENTRY:
            origin_idx = subsequence_begin_index + self.subsequence_length - 1
        else:
            raise ValueError(f"Unknown origin mode {self.origin_mode}")

        assert origin_idx >= 0 and origin_idx < len(
            sequence
        ), f"origin_idx must be >= 0 and < len(sequence), got {origin_idx} and {len(sequence)}"
        assert subsequence_begin_index >= 0 and subsequence_begin_index + self.subsequence_length <= len(
            sequence
        ), f"offset must be >= 0 and offset + self.subsequence_length <= len(sequence), got subsequence_begin_index {subsequence_begin_index} and len(sequence) {len(sequence)} for max sequence len {self.max_sequence_length} and a subsequence length {self.subsequence_length}"
        subsequence_lst = [
            sequence.load(subsequence_begin_index + i, origin_idx)
            for i in range(self.subsequence_length)
        ]
        return subsequence_lst


class VarLenSubsequenceSupervisedFlowDataset(VarLenSubsequenceRawDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__getitem__ = partial(
            SubsequenceSupervisedFlowDataset.__getitem__, self=self)


class VarLenSubsequenceSupervisedFlowSpecificSubsetDataset(
        VarLenSubsequenceSupervisedFlowDataset):

    def __init__(self,
                 sequence_loader,
                 subsequence_length: int,
                 origin_mode: Union[OriginMode, str],
                 subset_file: Path,
                 max_pc_points: int = 90000):
        super().__init__(sequence_loader, subsequence_length, origin_mode,
                         max_pc_points)
        subset_file = Path(subset_file)
        assert subset_file.exists(
        ), f"subset file {self.subset_file} does not exist"
        self.subset_list = self._parse_subset_file(subset_file)

    def _parse_subset_file(self, subset_file) -> List[Tuple[str, int]]:
        # Load each file line by line and extract tuple of (log_id, log_idx)
        with open(subset_file, 'r') as f:
            lines = f.readlines()
        res_list = []
        for line in lines:
            log_id, log_idx = line.split(",")
            res_list.append((log_id, int(log_idx)))
        return res_list

    def __len__(self):
        return len(self.subset_list)

    def _get_subsequence(self, index):
        assert index >= 0 and index < len(
            self
        ), f"index must be >= 0 and < len(self), got {index} and {len(self)}"
        log_id, log_idx = self.subset_list[index]
        sequence = self.sequence_loader.load_sequence(log_id)

        if self.origin_mode == OriginMode.FIRST_ENTRY:
            origin_idx = log_idx
        elif self.origin_mode == OriginMode.LAST_ENTRY:
            origin_idx = log_idx + self.subsequence_length - 1
        else:
            raise ValueError(f"Unknown origin mode {self.origin_mode}")

        subsequence_lst = [
            sequence.load(log_idx + i, origin_idx)
            for i in range(self.subsequence_length)
        ]

        # Special process the last entry in the subsequence because it does not have a flow but we still
        # want to use it for eval, so we need to shim in a flow of zeros and a pc_classes of -1
        e = subsequence_lst[-1]
        if e['relative_flowed_pc'] is None:
            e['relative_flowed_pc'] = e['relative_pc']
        if e['pc_classes'] is None:
            e['pc_classes'] = np.zeros(e['relative_pc'].points.shape[0]) * -1

        return subsequence_lst


class VarLenSubsequenceUnsupervisedFlowDataset(VarLenSubsequenceRawDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__getitem__ = partial(
            SubsequenceUnsupervisedFlowDataset.__getitem__, self=self)
