from dataloaders import BucketedSceneFlowInputSequence
from bucketed_scene_flow_eval.datasets import Argoverse2CausalSceneFlow
from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame
import pytest
from pointclouds import transform_pc
from pathlib import Path
import numpy as np
import torch

from bucketed_scene_flow_eval.datastructures import SE3


@pytest.fixture
def frame_list_tuple() -> tuple[list[TimeSyncedSceneFlowFrame], BucketedSceneFlowInputSequence]:
    sequence_length = 5
    dataset_idx = 0
    dataset = Argoverse2CausalSceneFlow(
        root_dir=Path("/tmp/argoverse2_small/val/"), subsequence_length=sequence_length
    )
    frame_list = dataset[dataset_idx]
    torch_sequence = BucketedSceneFlowInputSequence.from_frame_list(
        idx=dataset_idx, frame_list=frame_list, pc_max_len=120000, loader_type=dataset.loader_type()
    )

    return frame_list, torch_sequence


def test_transform_pc_simple():
    # Make a fake 1x3 pointcloud and transform it
    pc = np.array([[1, 2, 3]]).astype(np.float32)
    transform_se3 = SE3.from_rot_w_x_y_z_translation_x_y_z(
        rw=0.290, rx=0.556, ry=-0.361, rz=-0.690, tx=5, ty=6, tz=7
    )
    transformed_pc = transform_se3.transform_points(pc)

    pc_torch = torch.from_numpy(pc)

    transform_torch = torch.from_numpy(transform_se3.to_array().astype(np.float32))
    transformed_pc_torch = transform_pc(pc_torch, transform_torch)

    assert np.allclose(transformed_pc, transformed_pc_torch.numpy(), atol=1e-6)


def test_transform_pc_full_pc(
    frame_list_tuple: tuple[list[TimeSyncedSceneFlowFrame], BucketedSceneFlowInputSequence]
):
    frame_list, torch_sequence = frame_list_tuple
    assert len(frame_list) == len(torch_sequence)

    # Compare transform against the SE3 transforms from bucketed_scene_flow_eval.

    global_pc = frame_list[0].pc.global_pc
    global_pc_torch = torch_sequence.get_global_pc(0)

    assert (
        global_pc.shape == global_pc_torch.shape
    ), f"Expected shape {global_pc.shape}, got {global_pc_torch.shape}"

    # Check that the raw point values are allclose
    assert np.allclose(global_pc.points, global_pc_torch.numpy(), atol=1e-6)

    transform_se3 = SE3.from_rot_w_x_y_z_translation_x_y_z(
        rw=0.290, rx=0.556, ry=-0.361, rz=-0.690, tx=5, ty=6, tz=7
    )
    transform_torch = torch.from_numpy(transform_se3.to_array().astype(np.float32))

    transformed_global_pc = global_pc.transform(transform_se3)
    transformed_global_pc_torch = transform_pc(global_pc_torch, transform_torch)

    resudiual: np.ndarray = transformed_global_pc.points - transformed_global_pc_torch.numpy()

    assert np.allclose(
        transformed_global_pc.points, transformed_global_pc_torch.numpy(), atol=1e-5
    ), f"Expected transformed point clouds to be allclose; got biggest residual of {np.abs(resudiual).max()}"
