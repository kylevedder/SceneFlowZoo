import argparse
from pathlib import Path
import multiprocessing
from typing import Tuple, List
import tqdm
import shutil


# Argoverse 2 has 700 * 15 * 10 = 105,000 frames
# Argoverse Lidar has 20,000 * 30 * 10 = 6,000,000 frames
# Therefore we need ~6 frame pairs per trajectory.

# Get path to Argoverse Lidar dataset
parser = argparse.ArgumentParser()
parser.add_argument('lidar_path', type=Path)
parser.add_argument('output_folder', type=Path)
args = parser.parse_args()

assert args.lidar_path.is_dir(), f"Path {args.lidar_path} is not a directory"
# Check that the directory contains the expected number of sequences
sequence_folders = []
for subfolder in ["train", "val", "test"]:
    sequence_folders.extend((args.lidar_path / subfolder).glob("*"))
sequence_folders = sorted(sequence_folders, key=lambda x: x.name.lower())
if len(
    sequence_folders
) == 20000:
    print(f"WARNING: Expected 20000 sequences, found {len(sequence_folders)}")


def get_sequence_pairs(
        sequence_folder: Path,
        num_pairs: int = 6) -> List[Tuple[Path, Tuple[Path, Path]]]:
    """Get all pairs of frames in a sequence"""

    lidar_files = sorted(sequence_folder.glob("sensors/lidar/*.feather"))
    # We need num_pairs per trajectory
    chunk_step_size = (len(lidar_files) - 1) // (num_pairs - 1)
    frame_t = lidar_files[::chunk_step_size]
    frame_t1 = lidar_files[1::chunk_step_size]
    frame_pairs = list(zip(frame_t, frame_t1))
    return [(sequence_folder, frame_pair) for frame_pair in frame_pairs]


def save_sequence_pair(input: Tuple[Path, Tuple[Path, Path]]):
    """Create a sequence on disk using only the pair of frames. 
    
    Rather than copying any files, this will all be done via symlinks"""
    sequence_folder, frame_pair = input
    frame_t, frame_t1 = frame_pair

    output_sequence_folder = args.output_folder / (sequence_folder.name + "_" +
                                                   frame_t.stem)

    if output_sequence_folder.exists():
        shutil.rmtree(output_sequence_folder)
    # Create the output sequence folder
    output_sequence_folder.mkdir(exist_ok=True, parents=True)

    # Symlink the calibration, map, and SE3 files
    for subfolder in ["calibration", "map", "city_SE3_egovehicle.feather"]:
        subfolder_path = sequence_folder / subfolder
        output_subfolder_path = output_sequence_folder / subfolder
        output_subfolder_path.symlink_to(subfolder_path)

    # Symlink the lidar files
    output_lidar_folder = output_sequence_folder / "sensors/lidar"
    output_lidar_folder.mkdir(exist_ok=True, parents=True)
    output_lidar_t = output_lidar_folder / frame_t.name
    output_lidar_t.symlink_to(frame_t)
    output_lidar_t1 = output_lidar_folder / frame_t1.name
    output_lidar_t1.symlink_to(frame_t1)


print("Getting sequence pairs...")
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    folder_frame_lst = list(
        tqdm.tqdm(pool.imap_unordered(get_sequence_pairs, sequence_folders),
                  total=len(sequence_folders)))

# Flatten folder_frame_lst
folder_frame_lst = [item for sublist in folder_frame_lst for item in sublist]

print("Saving sequence pairs...")
with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    # pool.starmap(save_sequence_pair, folder_frame_lst)
    list(tqdm.tqdm(pool.imap_unordered(save_sequence_pair, folder_frame_lst), total=len(folder_frame_lst)))