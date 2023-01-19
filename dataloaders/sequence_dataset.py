import torch
import numpy as np
from tqdm import tqdm
import enum
from typing import Union


class OriginMode(enum.Enum):
    FIRST_ENTRY = 0
    LAST_ENTRY = 1


class SubsequenceDataset(torch.utils.data.Dataset):

    def __init__(self,
                 sequence_loader,
                 subsequence_length: int,
                 max_sequence_length: int,
                 origin_mode: Union[OriginMode, str],
                 max_pc_points: int = 90000,
                 shuffle=False):
        self.sequence_loader = sequence_loader
        assert subsequence_length > 0, f"subsequence_length must be > 0, got {subsequence_length}"
        assert max_sequence_length > 0, f"max_sequence_length must be > 0, got {max_sequence_length}"

        self.subsequence_length = subsequence_length

        if isinstance(origin_mode, str):
            origin_mode = OriginMode[origin_mode]
        self.origin_mode = origin_mode
        self.max_pc_points = max_pc_points

        self.subsequence_id_index = []
        for id in tqdm(sequence_loader.get_sequence_ids(),
                       desc="Preprocessing sequences",
                       leave=False):
            # sequence = sequence_loader.load_sequence(id)
            num_subsequences = max_sequence_length // subsequence_length
            assert num_subsequences > 0, f"num_subsequences must be > 0, got {num_subsequences}"
            self.subsequence_id_index.extend([
                (id, i) for i in range(num_subsequences)
            ])

        self.subsequence_id_shuffled_index = list(
            range(len(self.subsequence_id_index)))
        if shuffle:
            random_state = np.random.RandomState(len(
                self.subsequence_id_index))
            random_state.shuffle(self.subsequence_id_shuffled_index)

    def __len__(self):
        return len(self.subsequence_id_index)

    def __getitem__(self, index):
        shuffled_index = self.subsequence_id_shuffled_index[index]
        id, subsequence_index = self.subsequence_id_index[shuffled_index]
        sequence = self.sequence_loader.load_sequence(id)

        offset = subsequence_index * self.subsequence_length
        if self.origin_mode == OriginMode.FIRST_ENTRY:
            origin_idx = offset
        elif self.origin_mode == OriginMode.LAST_ENTRY:
            origin_idx = offset + self.subsequence_length - 1
        else:
            raise ValueError(f"Unknown origin mode {self.origin_mode}")
        subsequence_lst = [
            sequence.load(offset + i, origin_idx)
            for i in range(self.subsequence_length)
        ]
        pc_arrays = [
            pc.to_fixed_array(self.max_pc_points) for pc, _ in subsequence_lst
        ]
        pose_arrays = [pose.to_array() for _, pose in subsequence_lst]
        pc_array_stack = np.stack(pc_arrays, axis=0).astype(np.float32)
        pose_array_stack = np.stack(pose_arrays, axis=0).astype(np.float32)

        return {
            "pc_array_stack": pc_array_stack,
            "pose_array_stack": pose_array_stack,
            "data_index": index
        }
