import torch
import numpy as np
from tqdm import tqdm
import enum
from typing import Union, List, Tuple, Dict, Optional, Any
from pointclouds import to_fixed_array
from pathlib import Path


class OriginMode(enum.Enum):
    FIRST_ENTRY = 0
    LAST_ENTRY = 1


class SubsequenceRawDataset(torch.utils.data.Dataset):

    def __init__(self,
                 sequence_loader,
                 subsequence_length: int,
                 max_sequence_length: int,
                 origin_mode: Union[OriginMode, str],
                 max_pc_points: int = 90000,
                 subset_fraction: float = 1.0,
                 subset_mode='random',
                 shuffle=False):
        assert subset_mode in [
            'random', 'sequential'
        ], f"subset_mode must be 'random' or 'sequential', got {subset_mode}"
        assert 1.0 >= subset_fraction > 0.0, f"subset_fraction must be in (0.0, 1.0], got {subset_fraction}"
        self.sequence_loader = sequence_loader
        assert subsequence_length > 0, f"subsequence_length must be > 0, got {subsequence_length}"
        assert max_sequence_length > 0, f"max_sequence_length must be > 0, got {max_sequence_length}"

        self.subsequence_length = subsequence_length
        self.max_sequence_length = max_sequence_length

        if isinstance(origin_mode, str):
            origin_mode = OriginMode[origin_mode]
        self.origin_mode = origin_mode
        self.max_pc_points = max_pc_points

        self.subsequence_id_begin_index = []
        for id in sequence_loader.get_sequence_ids():
            num_subsequences = max_sequence_length - subsequence_length + 1
            assert num_subsequences > 0, f"num_subsequences must be > 0, got {num_subsequences}"
            self.subsequence_id_begin_index.extend([
                (id, i) for i in range(num_subsequences)
            ])

        self.subsequence_id_shuffled_index = list(
            range(len(self.subsequence_id_begin_index)))

        subsequence_max_index = int(
            len(self.subsequence_id_shuffled_index) * subset_fraction)

        if subset_fraction < 1.0 and subset_mode == 'sequential':
            print(
                f"Using only {subsequence_max_index} of {len(self.subsequence_id_shuffled_index)} sequences."
            )
            self.subsequence_id_shuffled_index = self.subsequence_id_shuffled_index[:
                                                                                    subsequence_max_index]

        if shuffle:
            random_state = np.random.RandomState(
                len(self.subsequence_id_begin_index))
            random_state.shuffle(self.subsequence_id_shuffled_index)

        if subset_fraction < 1.0 and subset_mode == 'random':
            print(
                f"Using only {subsequence_max_index} of {len(self.subsequence_id_shuffled_index)} sequences."
            )
            self.subsequence_id_shuffled_index = self.subsequence_id_shuffled_index[:
                                                                                    subsequence_max_index]

    def __len__(self):
        return len(self.subsequence_id_shuffled_index)

    def _get_subsequence(self, index):
        assert index >= 0 and index < len(
            self
        ), f"index must be >= 0 and < len(self), got {index} and {len(self)}"
        assert index < len(
            self.subsequence_id_shuffled_index
        ), f"index must be < len(self.subsequence_id_shuffled_index), got {index} and {len(self.subsequence_id_shuffled_index)}"
        shuffled_index = self.subsequence_id_shuffled_index[index]
        id, subsequence_begin_index = self.subsequence_id_begin_index[
            shuffled_index]
        sequence = self.sequence_loader.load_sequence(id)
        assert len(
            sequence
        ) >= self.max_sequence_length, f"actual len(sequence) must be >= promised self.max_sequence_length, got {len(sequence)} vs {self.max_sequence_length}"

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

    def _subsequence_lst_to_output_dict(self,
                                        subsequence_lst) -> Dict[str, Any]:
        pc_arrays = [
            e['relative_pc'].to_fixed_array(self.max_pc_points)
            for e in subsequence_lst
        ]
        pose_arrays = [e['relative_pose'].to_array() for e in subsequence_lst]
        pc_with_ground_arrays = [
            e['relative_pc_with_ground'].to_fixed_array(self.max_pc_points)
            for e in subsequence_lst
        ]
        is_ground_arrays = [
            to_fixed_array(e['is_ground_points'].astype(np.float32),
                           self.max_pc_points) for e in subsequence_lst
        ]
        log_ids = [e['log_id'] for e in subsequence_lst]
        log_idxes = [e['log_idx'] for e in subsequence_lst]
        pc_array_stack = np.stack(pc_arrays, axis=0).astype(np.float32)
        pose_array_stack = np.stack(pose_arrays, axis=0).astype(np.float32)
        pc_with_ground_array_stack = np.stack(pc_with_ground_arrays,
                                              axis=0).astype(np.float32)
        is_ground_array_stack = np.stack(is_ground_arrays,
                                         axis=0).astype(np.float32)

        return {
            "pc_array_stack": pc_array_stack,
            "pose_array_stack": pose_array_stack,
            "pc_with_ground_array_stack": pc_with_ground_array_stack,
            "is_ground_array_stack": is_ground_array_stack,
            "log_ids": log_ids,
            "log_idxes": log_idxes,
        }

    def __getitem__(self, index):
        assert index >= 0 and index < len(
            self
        ), f"index must be >= 0 and < len(self), got {index} and {len(self)}"

        subsequence_lst = self._get_subsequence(index)

        ret_dict = self._subsequence_lst_to_output_dict(subsequence_lst)
        ret_dict['data_index'] = index
        return ret_dict


class SubsequenceSupervisedFlowDataset(SubsequenceRawDataset):

    def _subsequence_lst_to_output_dict(self,
                                        subsequence_lst) -> Dict[str, Any]:
        ret_dict = super()._subsequence_lst_to_output_dict(subsequence_lst)
        flowed_pc_arrays = [
            e['relative_flowed_pc'].to_fixed_array(self.max_pc_points)
            for e in subsequence_lst
        ]
        pc_class_masks = [
            to_fixed_array(e['pc_classes'].astype(np.float32),
                           self.max_pc_points) for e in subsequence_lst
        ]
        flowed_pc_array_stack = np.stack(flowed_pc_arrays,
                                         axis=0).astype(np.float32)
        pc_class_mask_stack = np.stack(pc_class_masks,
                                       axis=0).astype(np.float32)
        ret_dict['flowed_pc_array_stack'] = flowed_pc_array_stack
        ret_dict['pc_class_mask_stack'] = pc_class_mask_stack
        return ret_dict


class SubsequenceSupervisedFlowSpecificSubsetDataset(
        SubsequenceSupervisedFlowDataset):

    def __init__(self,
                 sequence_loader,
                 subsequence_length: int,
                 max_sequence_length: int,
                 origin_mode: Union[OriginMode, str],
                 subset_file: Path,
                 max_pc_points: int = 90000):
        super().__init__(sequence_loader, subsequence_length,
                         max_sequence_length, origin_mode, max_pc_points)
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


class SubsequenceUnsupervisedFlowDataset(SubsequenceRawDataset):

    def _subsequence_lst_to_output_dict(self,
                                        subsequence_lst) -> Dict[str, Any]:
        ret_dict = super()._subsequence_lst_to_output_dict(subsequence_lst)

        def _squeeze_flow(flow: np.ndarray) -> np.ndarray:
            if flow.ndim == 3:
                assert flow.shape[
                    0] == 1, f"Flow must have 1 channel, got {flow.shape[0]}"
                return flow.squeeze(0)
            elif flow.ndim == 2:
                return flow
            else:
                raise ValueError(
                    f"Flow must have 2 or 3 dimensions, got {flow.ndim}")

        flow_arrays = [
            to_fixed_array(_squeeze_flow(e['flow']), self.max_pc_points)
            for e in subsequence_lst if e['flow'] is not None
        ]
        flow_array_stack = np.stack(flow_arrays, axis=0).astype(np.float32)
        ret_dict['flow_array_stack'] = flow_array_stack
        return ret_dict


class ConcatDataset(torch.utils.data.Dataset):
    r"""Dataset to concatenate multiple datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    def __init__(self, datasets: List[torch.utils.data.Dataset]):
        self.datasets = datasets
        for d in self.datasets:
            assert isinstance(
                d, torch.utils.data.Dataset
            ), f"ConcatDataset only supports datasets, got {type(d)}"
        self._length = sum(len(d) for d in self.datasets)
        print(
            f"Concatenated {len(self.datasets)} datasets with total length {self._length})"
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        assert idx >= 0, f"Index must be >= 0, got {idx}"
        assert idx < len(
            self), f"Index must be < len(self), got {idx} and {len(self)}"
        for d in self.datasets:
            if idx < len(d):
                return d[idx]
            idx -= len(d)
        raise IndexError('Index out of range')