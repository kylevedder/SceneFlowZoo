import argparse
import json
from pathlib import Path
from typing import Dict


def load_json(file_path: Path) -> Dict[str, int]:
    with file_path.open("r") as f:
        return json.load(f)


def validate_and_symlink_sequences(
    json_data: Dict[str, int], base_path: Path, target_dir: Path
) -> None:
    for sequence_name, sequence_len in sorted(json_data.items()):
        sequence_path = base_path / f"sequence_len_{sequence_len}" / sequence_name
        if not sequence_path.exists():
            raise ValueError(f"Invalid path: {sequence_path} does not exist")

        symlink_path = target_dir / sequence_name
        if symlink_path.exists():
            try:
                symlink_path.unlink()
            except Exception as e:
                raise RuntimeError(f"Failed to remove existing symlink: {symlink_path}") from e

        symlink_path.symlink_to(sequence_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate sequence paths from JSON and create symlinks."
    )
    parser.add_argument("json_path", type=Path, help="Path to the JSON file.")
    parser.add_argument("base_path", type=Path, help="Base directory containing the sequences.")
    parser.add_argument("target_dir", type=Path, help="Directory to create symlinks in.")
    args = parser.parse_args()

    assert args.base_path.exists(), f"Base path {args.base_path} does not exist."
    assert args.json_path.exists(), f"JSON path {args.json_path} does not exist."

    args.target_dir.mkdir(parents=True, exist_ok=True)

    json_data = load_json(args.json_path)
    validate_and_symlink_sequences(json_data, args.base_path, args.target_dir)


if __name__ == "__main__":
    main()
