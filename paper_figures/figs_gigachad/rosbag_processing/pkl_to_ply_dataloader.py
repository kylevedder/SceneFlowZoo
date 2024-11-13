from bucketed_scene_flow_eval.datasets import OrbbecAstra
from bucketed_scene_flow_eval.datastructures import (
    TimeSyncedSceneFlowFrame,
    ColoredSupervisedPointCloudFrame,
)
from pathlib import Path
import argparse
import numpy as np
import tqdm

# Parse args. Take as single argument Path to data directory.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("sequence_length", type=int)
    return parser.parse_args()


def create_dataset(data_dir: Path, sequence_length: int) -> list[TimeSyncedSceneFlowFrame]:
    dataset = OrbbecAstra(root_dir=data_dir, flow_dir=None, subsequence_length=sequence_length)
    seq = dataset[0]
    return seq


def write_ply(filename: Path, points: np.ndarray):
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
            # Only write 2 decimal places for x, y, z
            f.write(f"{x:.2f} {y:.2f} {z:.2f} {int(r*255)} {int(g*255)} {int(b*255)}\n")


def save_sequence(seq: list[TimeSyncedSceneFlowFrame], output_dir: Path):
    for i, frame in enumerate(tqdm.tqdm(seq)):
        pc: ColoredSupervisedPointCloudFrame = frame.pc

        mask = pc.mask
        masked_points = pc.global_pc
        masked_colors = pc.colors[mask]
        assert (
            masked_points.shape[0] == masked_colors.shape[0]
        ), f"{masked_points.shape} != {masked_colors.shape}"

        points = np.concatenate([masked_points, masked_colors], axis=1)

        ply_file = output_dir / f"{i:04d}.ply"
        write_ply(ply_file, points)


def main():
    args = parse_args()
    seq = create_dataset(args.data_dir, args.sequence_length)
    save_sequence(seq, args.data_dir)


if __name__ == "__main__":
    main()
