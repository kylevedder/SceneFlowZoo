import numpy as np

from typing import Union


def to_fixed_array(array: np.ndarray,
                   max_len: int,
                   pad_val=np.nan) -> np.ndarray:
    if len(array) > max_len:
        np.random.RandomState(len(array)).shuffle(array)
        sliced_pts = array[:max_len]
        return sliced_pts
    else:
        pad_tuples = [(0, max_len - len(array))]
        for _ in range(array.ndim - 1):
            pad_tuples.append((0, 0))
        return np.pad(array, pad_tuples, constant_values=pad_val)


def from_fixed_array(array: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:

    if len(array.shape) == 2:
        check_array = array[:, 0]
    elif len(array.shape) == 1:
        check_array = array
    else:
        raise ValueError(f'unknown array shape {array.shape}')
    if isinstance(array, np.ndarray):
        are_valid_points = np.logical_not(np.isnan(check_array))
        are_valid_points = are_valid_points.astype(bool)
    else:
        import torch
        are_valid_points = torch.logical_not(torch.isnan(check_array))
        are_valid_points = are_valid_points.bool()
    return array[are_valid_points]
