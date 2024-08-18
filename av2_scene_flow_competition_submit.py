import argparse
import json
from pathlib import Path
import zipfile
import tqdm
from typing import Any, Optional

from util_scripts.validation import load_sequence_name_to_size_map, validate_sequence


def solicit_is_supervised() -> bool:
    # Ask user if they are submitting a supervised or unsupervised model
    while True:
        user_input = input("Does your method use *any* labels from the AV2 dataset? (y/n): ")
        if user_input.lower() == "y":
            return True
        elif user_input.lower() == "n":
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def build_metadata(is_supervised: Optional[bool]) -> dict[str, Any]:
    return {
        "Is Supervised?": (is_supervised if is_supervised is not None else solicit_is_supervised()),
    }


def validate_and_archive_feather_files(
    metadata: dict[str, Any],
    root_folder_path: Path,
    sequence_to_size_map: dict[str, int],
    archive_path: Path,
):
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("metadata.json", json.dumps(metadata, indent=4))

        for subfolder in tqdm.tqdm(
            sorted(root_folder_path.iterdir()), desc="Validating and archiving folders"
        ):
            if subfolder.is_dir():  # Ensure it's a directory
                glob_pattern = "*.feather"
                validate_sequence(subfolder, glob_pattern, sequence_to_size_map)
                # Add every 5th .feather file to the zip, preserving the subfolder structure
                for i, feather_file in enumerate(sorted(subfolder.glob(glob_pattern))):
                    if i % 5 == 0:  # Add every 5th file
                        # Calculate path relative to the root folder to preserve structure
                        relative_path = feather_file.relative_to(root_folder_path)
                        save_relative_path = relative_path.parent / f"{i:010d}.feather"
                        zipf.write(feather_file, arcname=str(save_relative_path))
        print(f"Archive created at '{archive_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Validate the number of .feather files in subfolders against a validation JSON and archive every 5th file into a single zip, preserving the subfolder structure."
    )
    parser.add_argument("root_folder_path", type=Path, help="Path to the root folder.")
    parser.add_argument(
        "--validation_json",
        type=Path,
        default="data_prep_scripts/argo/av2_test_correct_num_flow_feathers.json",
        help="Path to the validation JSON file.",
    )
    parser.add_argument(
        "--archive_path", type=Path, help="Path to the output archive zip file.", default=None
    )
    # Argument to set if supervised
    parser.add_argument("--is_supervised", type=bool, help="Is the model supervised?", default=None)
    args = parser.parse_args()

    if args.archive_path is None:
        # Default archive path: root folder name + '_av2_2024_sf_submission.zip'
        default_archive_name = f"{args.root_folder_path.name}_av2_2024_sf_submission.zip"
        args.archive_path = args.root_folder_path.parent / default_archive_name

    # Load validation data
    validation_data = load_sequence_name_to_size_map(args.validation_json)

    # Remove archive if it already exists
    if args.archive_path.exists():
        args.archive_path.unlink()

    # Add metadata to the archive
    metadata = build_metadata(args.is_supervised)

    # Validate folder counts and archive every 5th feather file
    validate_and_archive_feather_files(
        metadata, args.root_folder_path, validation_data, args.archive_path
    )


if __name__ == "__main__":
    main()
