import argparse
from pathlib import Path


def get_all_sequence_folders(folder: Path) -> list[Path]:
    """
    Get all the sequence folders in the given data folder.
    """
    return sorted([x for x in folder.iterdir() if x.is_dir()])


def make_data_sequence_folder(
    input_data_sequence_folder: Path, data_feathers: list[Path], output_data_sequence_folder: Path
):
    # Ensure output folder exists
    output_data_sequence_folder.mkdir(parents=True, exist_ok=True)

    symlink_objects = ["annotations.feather", "calibration", "city_SE3_egovehicle.feather", "map"]
    for item in symlink_objects:
        src = input_data_sequence_folder / item
        dst = output_data_sequence_folder / item
        if src.is_dir():
            dst.symlink_to(src, target_is_directory=True)
        elif src.is_file():
            dst.symlink_to(src)

    data_feathers_parent = output_data_sequence_folder / "sensors/lidar"
    data_feathers_parent.mkdir(parents=True, exist_ok=True)
    for data_feather in data_feathers:
        dst = data_feathers_parent / data_feather.name
        dst.symlink_to(data_feather)


def make_flow_folder(data_feathers: list[Path], output_data_sequence_folder: Path):
    output_data_sequence_folder.mkdir(parents=True, exist_ok=True)
    for data_feather in data_feathers:
        dst = output_data_sequence_folder / data_feather.name
        dst.symlink_to(data_feather)


def process_sequence(
    input_data_sequence_folder: Path,
    input_flow_label_sequence_folder: Path,
    sequence_length: int,
    output_data_sequence_parent_folder: Path,
    output_flow_label_sequence_parent_folder: Path,
):

    data_feather_files = sorted(
        (input_data_sequence_folder / "sensors" / "lidar").glob("*.feather")
    )

    flow_feather_files = sorted(input_flow_label_sequence_folder.glob("*.feather"))

    assert len(data_feather_files) - 1 == len(flow_feather_files)

    num_chunks = len(flow_feather_files) // sequence_length

    # Iterate over each chunk
    for chunk_idx in range(num_chunks):
        chunk_data_feather_files = data_feather_files[
            chunk_idx * sequence_length : (chunk_idx + 1) * sequence_length
        ]
        chunk_flow_feather_files = flow_feather_files[
            chunk_idx * sequence_length : (chunk_idx + 1) * sequence_length - 1
        ]

        chunk_sequence_name = (
            input_data_sequence_folder.name + f"seqlen{sequence_length:03d}idx{chunk_idx:06d}"
        )

        chunk_output_data_sequence_folder = output_data_sequence_parent_folder / chunk_sequence_name
        output_flow_label_sequence_folder = (
            output_flow_label_sequence_parent_folder / chunk_sequence_name
        )

        make_data_sequence_folder(
            input_data_sequence_folder, chunk_data_feather_files, chunk_output_data_sequence_folder
        )
        make_flow_folder(chunk_flow_feather_files, output_flow_label_sequence_folder)


def main(input_data: Path, input_flow_labels: Path, sequence_length: int, output_root_folder: Path):

    input_data_subfolders = get_all_sequence_folders(input_data)
    input_flow_labels_subfolders = get_all_sequence_folders(input_flow_labels)

    output_data = output_root_folder / input_data.name
    output_flow_labels = output_root_folder / input_flow_labels.name

    # Ensure that the names all match exactly
    assert len(input_data_subfolders) == len(input_flow_labels_subfolders), (
        f"Number of subfolders in data folder ({len(input_data_subfolders)}) "
        f"does not match the number of subfolders in flow labels folder "
        f"({len(input_flow_labels_subfolders)})"
    )

    assert all(
        x.name == y.name for x, y in zip(input_data_subfolders, input_flow_labels_subfolders)
    ), "Subfolder names do not match"

    for input_data_sequence_folder, input_flow_label_sequence_folder in zip(
        input_data_subfolders, input_flow_labels_subfolders
    ):
        process_sequence(
            input_data_sequence_folder,
            input_flow_label_sequence_folder,
            sequence_length,
            output_data,
            output_flow_labels,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process data and flow labels.")

    parser.add_argument("input_data", type=Path, help="Path to the input data folder.")
    parser.add_argument(
        "input_flow_labels", type=Path, help="Path to the input flow labels folder."
    )
    parser.add_argument("sequence_length", type=int, help="Length of the sequences.")
    parser.add_argument("output_root_folder", type=Path, help="Path to the output root folder.")

    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.input_data, args.input_flow_labels, args.sequence_length, args.output_root_folder)
