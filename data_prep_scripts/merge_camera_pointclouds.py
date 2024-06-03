import argparse
from pathlib import Path

from bucketed_scene_flow_eval.datastructures import O3DVisualizer, PointCloud
from bucketed_scene_flow_eval.utils import load_feather, save_feather
import pandas as pd
import tqdm


def build_global_index_to_files_mapping(
    pc_folder: Path,
) -> dict[int, list[Path]]:
    """
    Creates a mapping of global frame indices to lists of corresponding feather files.
    """
    global_index_to_files: dict[int, list[Path]] = {}
    for offset_folder in pc_folder.iterdir():
        if not offset_folder.is_dir():
            continue
        offset_idx = int(offset_folder.name)
        for feather_file in offset_folder.glob("*.feather"):
            global_idx = offset_idx + int(feather_file.stem)
            if global_idx not in global_index_to_files:
                global_index_to_files[global_idx] = []
            global_index_to_files[global_idx].append(feather_file)
    return global_index_to_files


def merge_and_save_feather_files(
    global_index_to_files: dict[int, list[Path]], pc_folder: Path
) -> None:
    """
    Merges feather files for each global index and saves the result.
    """

    for global_idx, feather_files in sorted(global_index_to_files.items()):
        dfs = [load_feather(file, verbose=False) for file in sorted(feather_files)]
        merged_df = pd.concat(dfs, ignore_index=True)
        output_file = pc_folder / f"{global_idx:06d}.feather"
        save_feather(output_file, merged_df, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-frame .feather files into single-frame files."
    )
    parser.add_argument(
        "base_path", type=Path, help="Path to the base folder containing sequences."
    )
    args = parser.parse_args()

    sequence_folders = sorted([f for f in args.base_path.iterdir() if f.is_dir()])
    for sequence_folder in tqdm.tqdm(sequence_folders):
        camera_pc_folder = sequence_folder / "sensors" / "camera_pc"
        assert camera_pc_folder.is_dir(), f"Invalid sequence folder: {sequence_folder}"
        global_index_to_files = build_global_index_to_files_mapping(camera_pc_folder)
        merge_and_save_feather_files(global_index_to_files, camera_pc_folder)


if __name__ == "__main__":
    main()
