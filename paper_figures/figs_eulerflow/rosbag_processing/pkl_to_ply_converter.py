import pickle
import numpy as np
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import os


def write_ply(filename, points):
    """
    Write a PLY file from a point cloud data.
    Args:
        filename (str): Output PLY file path.
        points (numpy.ndarray): Nx6 array with x, y, z, r, g, b values.
    """
    with open(filename, "w") as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write point data
        for point in points:
            x, y, z, r, g, b = point
            f.write(f"{x} {y} {z} {int(r*255)} {int(g*255)} {int(b*255)}\n")


def convert_pkl_to_ply(pkl_file, ply_file):
    """
    Convert a PKL file containing point cloud data to a PLY file.
    Args:
        pkl_file (str): Path to input .pkl file.
        ply_file (str): Path to output .ply file.
    """
    # Load point cloud data from PKL file
    with open(pkl_file, "rb") as f:
        points = pickle.load(f)

    # Assuming the input is a list of lists or a 2D array of the form [[x, y, z, r, g, b], ...]
    points = np.array(points)  # Convert to numpy array if it's not already

    # Check if point cloud has correct shape (Nx6)
    if points.shape[1] != 6:
        raise ValueError("Input data must have exactly 6 columns: x, y, z, r, g, b")

    # Write to PLY
    write_ply(ply_file, points)


def process_single_file(pkl_file):
    """
    Convert a single .pkl file to .ply format.
    Args:
        pkl_file (Path): Path to the .pkl file.
    """
    # Construct the output .ply file path
    ply_file = pkl_file.with_suffix(".ply")

    # Convert the .pkl file to .ply
    print(f"Converting {pkl_file} to {ply_file}")
    convert_pkl_to_ply(pkl_file, ply_file)


def process_directory(input_dir, max_workers=None):
    """
    Process all .pkl files in a directory in parallel and convert them to .ply files.
    Args:
        input_dir (Path): Directory containing .pkl files.
        max_workers (int): Number of parallel workers to use. If None, use the number of CPUs.
    """
    input_dir = Path(input_dir)

    # Ensure the input is a directory
    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} is not a valid directory.")

    # Get all .pkl files in the directory
    pkl_files = sorted(input_dir.glob("*.pkl"))

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Map the process_single_file function to each .pkl file
        executor.map(process_single_file, pkl_files)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Convert PKL files to PLY files in a directory in parallel."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing PKL files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: number of CPUs).",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Process the directory in parallel, using default workers as number of CPUs if not specified
    process_directory(args.input_dir, max_workers=args.workers)
