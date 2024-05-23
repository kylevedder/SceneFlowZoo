import pytest
from dataloaders import TorchFullFrameDataset, TorchFullFrameInputSequence
from pathlib import Path
import torch


@pytest.fixture
def first_split_dataset() -> TorchFullFrameDataset:
    dataset = TorchFullFrameDataset(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=Path("/tmp/argoverse2_small/val/"),
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=4,
        split=dict(split_idx=0, num_splits=3),
    )

    return dataset


@pytest.fixture
def second_split_dataset() -> TorchFullFrameDataset:
    dataset = TorchFullFrameDataset(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=Path("/tmp/argoverse2_small/val/"),
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=4,
        split=dict(split_idx=1, num_splits=3),
    )

    return dataset


@pytest.fixture
def third_split_dataset() -> TorchFullFrameDataset:
    dataset = TorchFullFrameDataset(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=Path("/tmp/argoverse2_small/val/"),
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=4,
        split=dict(split_idx=2, num_splits=3),
    )

    return dataset


@pytest.fixture
def full_dataset() -> TorchFullFrameDataset:
    dataset = TorchFullFrameDataset(
        dataset_name="Argoverse2CausalSceneFlow",
        root_dir=Path("/tmp/argoverse2_small/val/"),
        flow_data_path=Path("/tmp/argoverse2_small/val_sceneflow_feather/"),
        subsequence_length=4,
    )

    return dataset


def test_load_first_split_datset(first_split_dataset: TorchFullFrameDataset):
    assert (
        len(first_split_dataset) == 104
    ), f"Expected 104 sequences, but got {len(first_split_dataset)}."


def test_load_second_split_datset(second_split_dataset: TorchFullFrameDataset):
    assert (
        len(second_split_dataset) == 103
    ), f"Expected 103 sequences, but got {len(second_split_dataset)}."


def test_load_third_split_datset(third_split_dataset: TorchFullFrameDataset):
    assert (
        len(third_split_dataset) == 103
    ), f"Expected 103 sequences, but got {len(third_split_dataset)}."


def test_load_full_datset(full_dataset: TorchFullFrameDataset):
    assert len(full_dataset) == 310, f"Expected 310 sequences, but got {len(full_dataset)}."


def test_load_full_and_first_split_get_same_elements(
    full_dataset: TorchFullFrameDataset, first_split_dataset: TorchFullFrameDataset
):
    full_elem = full_dataset[0]
    split_elem = first_split_dataset[0]

    assert len(full_elem) == len(
        split_elem
    ), f"Expected same length, but got {len(full_elem)} and {len(split_elem)}."

    for idx in range(len(full_elem)):
        assert torch.all(full_elem.get_full_global_pc(idx) == split_elem.get_full_global_pc(idx))


def test_load_full_and_second_split_get_same_elements(
    full_dataset: TorchFullFrameDataset, second_split_dataset: TorchFullFrameDataset
):
    full_elem = full_dataset[104]  # length of first split
    split_elem = second_split_dataset[0]

    assert len(full_elem) == len(
        split_elem
    ), f"Expected same length, but got {len(full_elem)} and {len(split_elem)}."

    for idx in range(len(full_elem)):
        assert torch.all(full_elem.get_full_global_pc(idx) == split_elem.get_full_global_pc(idx))


def test_load_full_and_third_split_get_same_elements(
    full_dataset: TorchFullFrameDataset, third_split_dataset: TorchFullFrameDataset
):
    full_elem = full_dataset[207]  # length of the first + second split, 104 + 103
    split_elem = third_split_dataset[0]

    assert len(full_elem) == len(
        split_elem
    ), f"Expected same length, but got {len(full_elem)} and {len(split_elem)}."

    for idx in range(len(full_elem)):
        assert torch.all(full_elem.get_full_global_pc(idx) == split_elem.get_full_global_pc(idx))
