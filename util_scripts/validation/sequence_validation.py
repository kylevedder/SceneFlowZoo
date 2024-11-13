from pathlib import Path
from typing import Optional
from bucketed_scene_flow_eval.utils import load_json, load_by_extension
import numpy as np


def load_sequence_name_to_size_map(validation_file_path: Path) -> dict[str, int]:
    data = load_json(validation_file_path)

    # Validate types
    assert isinstance(data, dict), f"Invalid data type {type(data)}."
    for key, value in data.items():
        assert isinstance(key, str), f"Invalid key type {type(key)}."
        assert isinstance(value, int), f"Invalid value type {type(value)}."

    return data


def load_sequence_name_to_pc_dims_map(validation_file_path: Path) -> dict[str, list[int]]:
    data = load_json(validation_file_path)

    # Validate types
    assert isinstance(data, dict), f"Invalid data type {type(data)}."
    for key, value in data.items():
        assert isinstance(key, str), f"Invalid key type {type(key)}."
        assert isinstance(value, list), f"Invalid value type {type(value)}."
        for dim in value:
            assert isinstance(dim, int), f"Invalid value type {type(value)}."

    return data


def _validate_pc_dims(
    sequence_folder: Path, glob_pattern: str, sequence_name_to_pc_dims_map: dict[str, list[int]]
):
    feather_files = sorted(sequence_folder.glob(glob_pattern))
    sequence_name = sequence_folder.name

    found_feather_file_lengths = [
        len(load_by_extension(feather_file, verbose=False)) for feather_file in feather_files
    ]
    expected_feather_file_lengths = sequence_name_to_pc_dims_map[sequence_name]

    # Ensure that the lists are at least the same length
    assert len(found_feather_file_lengths) == len(
        expected_feather_file_lengths
    ), f"LENGTH MISMATCH: Found feather file lengths {found_feather_file_lengths} do not match expected lengths {expected_feather_file_lengths}"

    expected_lens = np.array(expected_feather_file_lengths)
    found_lens = np.array(found_feather_file_lengths)

    num_matches = np.sum(expected_lens == found_lens)
    num_not_matches = np.sum(expected_lens != found_lens)

    assert (
        found_feather_file_lengths == expected_feather_file_lengths
    ), f"CONTENT MISMATCH: found {num_matches} matches and {num_not_matches} mismatches\n\nFOUND:\n{found_feather_file_lengths}\n\nEXPECTED:\n{expected_feather_file_lengths}"


def _validate_sequence_length(
    sequence_folder: Path, glob_pattern: str, sequence_name_to_size_map: dict[str, int]
):
    feather_files = sorted(sequence_folder.glob(glob_pattern))
    sequence_name = sequence_folder.name

    found_count = len(feather_files)
    expected_count = sequence_name_to_size_map[sequence_name]

    assert (
        found_count == expected_count
    ), f"COUNT MISMATCH: Found {found_count} files, expected {expected_count} for sequence {sequence_name}. Looking with pattern {sequence_folder}*{glob_pattern}"


def validate_sequence(
    sequence_folder: Path,
    glob_pattern: str,
    sequence_name_to_size_map: dict[str, int],
    sequence_name_to_pc_dims_map: Optional[dict[str, list[int]]] = None,
) -> None:

    if sequence_name_to_pc_dims_map is not None:
        _validate_pc_dims(sequence_folder, glob_pattern, sequence_name_to_pc_dims_map)
    _validate_sequence_length(sequence_folder, glob_pattern, sequence_name_to_size_map)
