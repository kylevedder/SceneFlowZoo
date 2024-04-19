import argparse
from pathlib import Path


def flatten_sequence_folder(
    non_causal_sequence: Path, output_folder: Path, glob_pattern: str = "*.feather"
):
    output_folder.mkdir(parents=True, exist_ok=True)

    # Non-causal sequence folder contains subfolders for each non-causal subsequence
    subsequence_folders = sorted(
        [subfolder for subfolder in non_causal_sequence.iterdir() if subfolder.is_dir()]
    )

    subsequence_files_list = [
        sorted(subsequence_folder.glob(glob_pattern)) for subsequence_folder in subsequence_folders
    ]

    # Ensure that all subsequence folders have the same length
    subsequence_files_lengths = [len(files) for files in subsequence_files_list]
    assert all(
        length == subsequence_files_lengths[0] for length in subsequence_files_lengths
    ), f"All subsequence folders must have the same length; instead got {subsequence_files_lengths}."

    running_counter = 0
    for subsequence_idx, subsequence_files in enumerate(subsequence_files_list):
        for source_file in subsequence_files:
            target_file = output_folder / f"{running_counter:010d}.feather"
            # Symlink to point the target file to the source file
            target_file.symlink_to(source_file)
            running_counter += 1

        # Make a dummy file to handle the gap between subsequences
        target_file = output_folder / f"{running_counter:010d}.feather"
        target_file.touch()
        running_counter += 1


def flatten_root_folder(non_causal_root: Path, output_root: Path):

    sequence_folders = sorted(
        [
            sequence_folder
            for sequence_folder in non_causal_root.iterdir()
            if sequence_folder.is_dir()
        ]
    )
    for sequence_folder in sequence_folders:

        # Flatten the sequence folder
        output_folder = output_root / sequence_folder.name
        flatten_sequence_folder(sequence_folder, output_folder)


if __name__ == "__main__":
    # Take path to the non-causal flow files stored in a folder and flatten them into a single folder
    parser = argparse.ArgumentParser(
        description="Flatten non-causal flow files into a single folder"
    )
    parser.add_argument(
        "non_causal_flow_folder", type=Path, help="Path to the non-causal flow folder"
    )
    parser.add_argument("output_folder", type=Path, help="Path to the output folder")
    args = parser.parse_args()

    flatten_root_folder(args.non_causal_flow_folder, args.output_folder)
