import numpy as np
from bucketed_scene_flow_eval.datastructures import PointCloud


def to_fixed_array_np(
    array: np.ndarray, max_len: int, pad_val=np.nan, allow_pc_slicing: bool = False
) -> np.ndarray:
    assert isinstance(array, np.ndarray), f"Expected numpy array, got {type(array)}"
    if len(array) > max_len:
        assert (
            allow_pc_slicing
        ), f"Point cloud is too large ({len(array)} > {max_len}), but allow_pc_slicing is False."
        np.random.RandomState(len(array)).shuffle(array)
        sliced_pts = array[:max_len]
        return sliced_pts
    else:
        pad_tuples = [(0, max_len - len(array))]
        for _ in range(array.ndim - 1):
            pad_tuples.append((0, 0))
        array = array.astype(np.float32)
        return np.pad(array, pad_tuples, constant_values=pad_val)


def to_fixed_array_torch(tensor, max_len, pad_val=float("nan"), allow_pc_slicing: bool = False):
    import torch

    assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"

    tensor = tensor.float()
    original_len = tensor.shape[0]

    if original_len > max_len:
        assert (
            allow_pc_slicing
        ), f"Point cloud is too large ({original_len} > {max_len}), but allow_pc_slicing is False."
        indices = torch.randperm(original_len)[:max_len]
        sliced_tensor = tensor[indices]
        return sliced_tensor
    else:
        # Determine padding based on tensor dimensionality
        if tensor.dim() == 1:  # For 1D tensors
            pad_size = max_len - original_len
            pad_tensor = torch.nn.functional.pad(
                tensor, (0, pad_size), mode="constant", value=pad_val
            )
        elif tensor.dim() == 2:  # For 2D tensors
            pad_size = max_len - original_len
            # Padding for 2D tensor requires specifying padding for both dimensions,
            # but we only pad along the first dimension (number of points)
            pad_tensor = torch.nn.functional.pad(
                tensor, (0, 0, 0, pad_size), mode="constant", value=pad_val
            )
        else:
            raise ValueError(
                "Tensor dimensionality not supported. Only 1D and 2D tensors are supported."
            )

        return pad_tensor


def from_fixed_array_valid_mask_np(array: np.ndarray) -> np.ndarray:
    assert isinstance(array, np.ndarray), f"Expected numpy array, got {type(array)}"
    if len(array.shape) == 2:
        # If it's an Nx3 array like a point cloud, all three dimensions are the
        # same valid / invalid state, so grab the first channel.
        check_array = array[:, 0]
    elif len(array.shape) == 1:
        check_array = array
    else:
        raise ValueError(
            f"unknown array shape {array.shape} -- can only handle (N, 3) or (N,) arrays"
        )

    are_valid_points = np.logical_not(np.isnan(check_array))
    are_valid_points = are_valid_points.astype(bool)

    return are_valid_points


def from_fixed_array_valid_mask_torch(array: "torch.Tensor") -> "torch.Tensor":
    import torch

    assert isinstance(array, torch.Tensor), f"Expected torch tensor, got {type(array)}"
    if len(array.shape) == 2:
        # If it's an Nx3 array like a point cloud, all three dimensions are the
        # same valid / invalid state, so grab the first channel.
        check_array = array[:, 0]
    elif len(array.shape) == 1:
        check_array = array
    else:
        raise ValueError(
            f"unknown array shape {array.shape} -- can only handle (N, 3) or (N,) arrays"
        )

    are_valid_points = torch.logical_not(torch.isnan(check_array))
    are_valid_points = are_valid_points.bool()
    return are_valid_points


def from_fixed_array_np(array: np.ndarray) -> np.ndarray:
    assert isinstance(array, np.ndarray), f"Expected numpy array, got {type(array)}"
    are_valid_points = from_fixed_array_valid_mask_np(array)
    return array[are_valid_points]


def from_fixed_array_torch(array: "torch.Tensor") -> "torch.Tensor":
    import torch

    assert isinstance(array, torch.Tensor), f"Expected torch tensor, got {type(array)}"
    are_valid_points = from_fixed_array_valid_mask_torch(array)
    return array[are_valid_points]


def transform_pc(pc: "torch.Tensor", transform: "torch.Tensor") -> "torch.Tensor":
    import torch

    """
    Transform an Nx3 point cloud by a 4x4 transformation matrix.
    """

    homogenious_pc = torch.cat((pc, torch.ones((pc.shape[0], 1), device=pc.device)), dim=1)
    homogenious_pc = homogenious_pc @ transform.T
    return homogenious_pc[:, :3]


def global_to_ego_flow(
    global_full_pc: "torch.Tensor",
    global_warped_full_pc: "torch.Tensor",
    global_to_ego: "torch.Tensor",
) -> "torch.Tensor":
    import torch

    ego_full_pc0 = transform_pc(global_full_pc, global_to_ego)
    ego_warped_full_pc0 = transform_pc(global_warped_full_pc, global_to_ego)

    return ego_warped_full_pc0 - ego_full_pc0
