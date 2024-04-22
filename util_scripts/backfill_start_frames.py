import argparse
from pathlib import Path
from util_scripts.validation import (
    load_sequence_name_to_pc_dims_map,
    validate_sequence,
)
import shutil
import tqdm


def validate_complete_root_folder(
    root_folder: Path,
    sequence_to_pc_dims_lst_map: dict[str, list[int]],
    glob_pattern: str = "*.feather",
):

    print(f"Validating root folder: {root_folder}")
    subfolders = sorted([subfolder for subfolder in root_folder.iterdir() if subfolder.is_dir()])
    subfolder_names = set(subfolder.name for subfolder in subfolders)

    # Ensure that all validation data keys are present in the subfolders
    missing_subfolders = set(sequence_to_pc_dims_lst_map.keys()) - subfolder_names
    assert len(missing_subfolders) == 0, f"Missing subfolders: {missing_subfolders}"

    sequence_to_len_map = {key: len(value) for key, value in sequence_to_pc_dims_lst_map.items()}

    for subfolder in tqdm.tqdm(subfolders):
        validate_sequence(subfolder, glob_pattern, sequence_to_len_map, sequence_to_pc_dims_lst_map)


def rename_files_in_folder(folder: Path, start_index: int, glob_pattern: str = "*.feather"):
    # First rename all the files in the folder to have an "old_" prefix
    for i, feather_file in enumerate(sorted(folder.glob(glob_pattern))):
        old_name = folder / f"old_{i:010d}.feather"
        feather_file.rename(old_name)

    # Now rename them back to the original names, starting from start_index
    for i, feather_file in enumerate(sorted(folder.glob(glob_pattern))):
        new_name = folder / f"{start_index + i:010d}.feather"
        feather_file.rename(new_name)


def get_first_n_files_from_folder(folder: Path, n: int, glob_pattern: str = "*.feather"):
    files = sorted(folder.glob(glob_pattern))
    return files[:n]


def backfill_from_complete_folder(
    complete_root_folder: Path,
    partial_root_folder: Path,
    sequence_to_pc_dims_lst_map: dict[str, list[int]],
    glob_pattern: str = "*.feather",
):
    # Ensure that the complete root folder and the partial root folder have the same subfolders
    complete_subfolders = sorted(
        [subfolder for subfolder in complete_root_folder.iterdir() if subfolder.is_dir()]
    )
    partial_subfolders = sorted(
        [subfolder for subfolder in partial_root_folder.iterdir() if subfolder.is_dir()]
    )

    complete_subfolder_names = set(subfolder.name for subfolder in complete_subfolders)
    partial_subfolder_names = set(subfolder.name for subfolder in partial_subfolders)

    assert (
        complete_subfolder_names == partial_subfolder_names
    ), "Subfolders do not match. This means an entire sequence is missing from the partial root folder."

    # For each partial subfolder, compute the number of missing files.
    missing_files_per_folder = [
        len(sequence_to_pc_dims_lst_map[subfolder.name]) - len(list(subfolder.glob(glob_pattern)))
        for subfolder in partial_subfolders
    ]

    # Ensure that all subfolders have the same number of missing files
    assert (
        len(set(missing_files_per_folder)) == 1
    ), f"Subfolders have different number of missing files: {set(missing_files_per_folder)}"

    num_files_to_copy = missing_files_per_folder[0]
    print(
        f"Copying first {num_files_to_copy} files from the complete root folder to the partial root folder"
    )

    for complete_subfolder, partial_subfolder in tqdm.tqdm(
        list(zip(complete_subfolders, partial_subfolders))
    ):
        # Rename existing files in the partial subfolder
        rename_files_in_folder(partial_subfolder, num_files_to_copy)
        # Copy the first n files from the complete subfolder to the partial subfolder, named from 0 to n-1
        first_n_files = get_first_n_files_from_folder(complete_subfolder, num_files_to_copy)
        for i, file in enumerate(first_n_files):
            new_name = partial_subfolder / f"{i:010d}.feather"
            shutil.copy(file, new_name)


def main():
    parser = argparse.ArgumentParser(
        description="Validate the number of .feather files in subfolders against a validation JSON and archive every 5th file into a single zip, preserving the subfolder structure."
    )
    parser.add_argument("partial_root_folder", type=Path, help="Path to the partial root folder.")
    parser.add_argument("complete_root_folder", type=Path, help="Path to the complete root folder.")
    parser.add_argument("output_root_folder", type=Path, help="Path to the output root folder.")
    parser.add_argument(
        "--sequence_pc_sizes_json",
        type=Path,
        default="data_prep_scripts/argo/av2_test_sizes.json",
        help="Path to the validation JSON file.",
    )

    args = parser.parse_args()

    # Load validation data
    sequence_to_pc_dims_lst_map = load_sequence_name_to_pc_dims_map(args.sequence_pc_sizes_json)

    # Validate the complete root folder, as that is the one that contains all the data
    validate_complete_root_folder(args.complete_root_folder, sequence_to_pc_dims_lst_map)

    # Delete the output folder if it already exists
    if args.output_root_folder.exists():
        print(f"Deleting existing output folder: {args.output_root_folder}")
        shutil.rmtree(args.output_root_folder)

    print("Copying partial root folder to output folder...")

    # Copy the partial root folder to the output folder
    shutil.copytree(args.partial_root_folder, args.output_root_folder)

    print("Copying complete. Starting backfill...")

    backfill_from_complete_folder(
        args.complete_root_folder, args.output_root_folder, sequence_to_pc_dims_lst_map
    )

    print("Validating backfilled folder...")

    validate_complete_root_folder(args.output_root_folder, sequence_to_pc_dims_lst_map)

    print(f"Backfill of {args.output_root_folder} complete.")


if __name__ == "__main__":
    main()
