import torch
import numpy as np


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, sequence_loader, subsequence_length: int):
        self.sequence_loader = sequence_loader
        assert subsequence_length > 0, f"subsequence_length must be > 0, got {subsequence_length}"

        self.subsequence_length = subsequence_length

        self.subsequence_id_index = []
        for id in sequence_loader.get_sequence_ids():
            sequence = sequence_loader.get_sequence(id)
            num_subsequences = len(sequence) // subsequence_length
            assert num_subsequences > 0, f"num_subsequences must be > 0, got {num_subsequences}"
            self.subsequence_id_index.extend([
                (id, i) for i in range(num_subsequences)
            ])

    def __len__(self):
        return len(self.subsequence_id_index)

    def __getitem__(self, index):
        id, subsequence_index = self.subsequence_id_index[index]
        sequence = self.sequence_loader.get_sequence(id)

        offset = subsequence_index * self.subsequence_length
        subsequence_lst = [
            sequence[offset + i] for i in range(self.subsequence_length)
        ]
        return subsequence_lst
