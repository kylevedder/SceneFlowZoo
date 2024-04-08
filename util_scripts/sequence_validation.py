import argparse
import json
from pathlib import Path
import zipfile
import tqdm
from typing import Any, Optional
from bucketed_scene_flow_eval.utils import load_json


def load_sequence_name_to_size_map(validation_file_path: Path) -> dict[str, int]:
    data = load_json(validation_file_path)

    # Validate types
    assert isinstance(data, dict), f"Invalid data type {type(data)}."
    for key, value in data.items():
        assert isinstance(key, str), f"Invalid key type {type(key)}."
        assert isinstance(value, int), f"Invalid value type {type(value)}."

    return data


def validate_sequence(
    sequence_folder: Path, glob_pattern: str, sequence_name_to_size_map: dict[str, int]
) -> None:
    assert sequence_folder.is_dir(), f"Invalid sequence folder {sequence_folder}."
    sequence_name = sequence_folder.name
    expected_count = sequence_name_to_size_map.get(sequence_name)
    if expected_count is None:
        raise ValueError(f"No validation data for '{sequence_name}'.")

    feather_files = sorted(sequence_folder.glob(glob_pattern))
    actual_count = len(feather_files)

    if actual_count != expected_count:
        raise ValueError(
            f"Validation failed for '{sequence_name}': expected {expected_count}, found {actual_count}"
        )
