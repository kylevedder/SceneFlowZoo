import argparse
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
import numpy as np


def process_file(file_path: Path, source_root: Path, target_root: Path, speed_threshold: float):
    # Load feather file
    df = pd.read_feather(file_path)

    # Compute the L2 norm for the flow vectors
    flow_vectors = df[["flow_tx_m", "flow_ty_m", "flow_tz_m"]].values
    speeds = np.linalg.norm(flow_vectors, axis=1)

    # Mask rows where the speed exceeds the threshold
    exceed_threshold = speeds > speed_threshold
    df.loc[exceed_threshold, ["flow_tx_m", "flow_ty_m", "flow_tz_m"]] = 0

    # Create corresponding subdirectory structure in the target directory
    relative_path = file_path.relative_to(source_root)
    target_file_path = target_root / relative_path
    target_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the modified dataframe back as a feather file
    df.reset_index(drop=True).to_feather(target_file_path)


def process_directory(dir_path: Path, source_root: Path, target_root: Path, speed_threshold: float):
    # Get all feather files in directory
    feather_files = list(dir_path.glob("*.feather"))

    # Process each feather file in the directory
    for file_path in feather_files:
        process_file(file_path, source_root, target_root, speed_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process feather files by L2 norm thresholding.")
    parser.add_argument(
        "source_root", type=Path, help="Root directory containing subfolders with feather files."
    )
    parser.add_argument("target_root", type=Path, help="Target root directory for processed files.")
    parser.add_argument(
        "speed_threshold", type=float, help="Speed threshold for filtering flow vectors."
    )

    args = parser.parse_args()

    # Ensure the target root exists
    args.target_root.mkdir(parents=True, exist_ok=True)

    # Get all subdirectories in the source root
    subdirectories = [d for d in args.source_root.iterdir() if d.is_dir()]

    # Process each directory in parallel
    Parallel(n_jobs=-1)(
        delayed(process_directory)(
            dir_path, args.source_root, args.target_root, args.speed_threshold
        )
        for dir_path in subdirectories
    )
