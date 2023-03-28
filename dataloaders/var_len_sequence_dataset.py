import torch
import numpy as np
from tqdm import tqdm
import enum
from typing import Union, List, Tuple, Dict, Optional, Any
from pointclouds import to_fixed_array


class OriginMode(enum.Enum):
    FIRST_ENTRY = 0
    LAST_ENTRY = 1


class VarLenSubsequenceRawDataset(torch.utils.data.Dataset):

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

    def __getitem__(self, index):
        assert index >= 0 and index < len(
            self
        ), f"index must be >= 0 and < len(self), got {index} and {len(self)}"

        subsequence_lst = self._get_subsequence(index)

        pc_arrays = [
            e['relative_pc'].to_fixed_array(self.max_pc_points)
            for e in subsequence_lst
        ]
        pose_arrays = [e['relative_pose'].to_array() for e in subsequence_lst]
        log_ids = [e['log_id'] for e in subsequence_lst]
        log_idxes = [e['log_idx'] for e in subsequence_lst]
        pc_array_stack = np.stack(pc_arrays, axis=0).astype(np.float32)
        pose_array_stack = np.stack(pose_arrays, axis=0).astype(np.float32)

        return {
            "pc_array_stack": pc_array_stack,
            "pose_array_stack": pose_array_stack,
            "data_index": index,
            "log_ids": log_ids,
            "log_idxes": log_idxes,
        }


class VarLenSubsequenceSupervisedFlowDataset(VarLenSubsequenceRawDataset):

    def __getitem__(self, index):
        subsequence_lst = self._get_subsequence(index)

        pc_arrays = [
            e['relative_pc'].to_fixed_array(self.max_pc_points)
            for e in subsequence_lst
        ]
        pose_arrays = [e['relative_pose'].to_array() for e in subsequence_lst]
        flowed_pc_arrays = [
            e['relative_flowed_pc'].to_fixed_array(self.max_pc_points)
            for e in subsequence_lst
        ]
        pc_class_masks = [
            to_fixed_array(e['pc_classes'].astype(np.float32),
                           self.max_pc_points) for e in subsequence_lst
        ]
        log_ids = [e['log_id'] for e in subsequence_lst]
        log_idxes = [e['log_idx'] for e in subsequence_lst]

        pc_array_stack = np.stack(pc_arrays, axis=0).astype(np.float32)
        pose_array_stack = np.stack(pose_arrays, axis=0).astype(np.float32)
        flowed_pc_array_stack = np.stack(flowed_pc_arrays,
                                         axis=0).astype(np.float32)
        pc_class_mask_stack = np.stack(pc_class_masks,
                                       axis=0).astype(np.float32)

        return {
            "pc_array_stack": pc_array_stack,
            "pose_array_stack": pose_array_stack,
            "flowed_pc_array_stack": flowed_pc_array_stack,
            "pc_class_mask_stack": pc_class_mask_stack,
            "data_index": index,
            "log_ids": log_ids,
            "log_idxes": log_idxes
        }


class VarLenSubsequenceUnsupervisedFlowDataset(VarLenSubsequenceRawDataset):

    def __getitem__(self, index):
        subsequence_lst = self._get_subsequence(index)

        pc_arrays = [
            e['relative_pc'].to_fixed_array(self.max_pc_points)
            for e in subsequence_lst
        ]
        pose_arrays = [e['relative_pose'].to_array() for e in subsequence_lst]

        flow_arrays = [
            to_fixed_array(e['flow'].squeeze(0), self.max_pc_points)
            for e in subsequence_lst
        ]
        log_ids = [e['log_id'] for e in subsequence_lst]
        log_idxes = [e['log_idx'] for e in subsequence_lst]

        pc_array_stack = np.stack(pc_arrays, axis=0).astype(np.float32)
        pose_array_stack = np.stack(pose_arrays, axis=0).astype(np.float32)
        flow_array_stack = np.stack(flow_arrays, axis=0).astype(np.float32)

        return {
            "pc_array_stack": pc_array_stack,
            "pose_array_stack": pose_array_stack,
            "flow_array_stack": flow_array_stack,
            "data_index": index,
            "log_ids": log_ids,
            "log_idxes": log_idxes
        }
