import argparse
from pathlib import Path

from bucketed_scene_flow_eval.datastructures import O3DVisualizer, PointCloud
from bucketed_scene_flow_eval.utils import load_npz, load_pickle, save_pickle
import numpy as np
import cv2


def orbbec_astra_disparity_to_point_cloud(disparity: np.ndarray) -> PointCloud:
    # Image is (512, 640)
    height, width = disparity.shape

    fx = 535.4  # Camera.fx
    fy = 539.2  # Camera.fy
    # Center of the image
    cx = 320.1  # Camera.cx
    cy = 247.6  # Camera.cy

    # Create meshgrid for pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Convert disparity to depth
    # We'll use a scaling factor here. You might need to adjust this based on your specific data.
    scale_factor = 1.0  # This value may need tuning
    epsilon = 1e-10  # Small value to avoid division by zero
    z = scale_factor / (disparity + epsilon)

    # Clip depth to a reasonable range (e.g., 0 to 7 meters)
    z = np.clip(z, 0, 7)

    # Calculate 3D coordinates
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    # Stack the coordinates
    points = np.stack((x, y, z), axis=-1)

    # Reshape to a list of points
    point_cloud = points.reshape(-1, 3)

    # Remove points with invalid depth
    point_cloud = point_cloud[point_cloud[:, 2] > 0]

    return PointCloud(point_cloud)


def load_gt_point_clouds(gt_folder: Path) -> list[PointCloud]:
    pkl_files = sorted(gt_folder.glob("*.pkl"))
    pkl_points = [load_pickle(pkl_file)[:, :3] for pkl_file in pkl_files]
    return [PointCloud(points) for points in pkl_points]


def load_frames_from_mp4(mp4_file: Path) -> np.ndarray:
    # Load the video
    cap = cv2.VideoCapture(str(mp4_file))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.stack(frames)


def save_pkl_pointclouds(
    rgb_frame: np.ndarray, point_cloud: PointCloud, save_folder: Path, idx: int
) -> None:

    # Ensure the rgb frame is in the correct format
    assert rgb_frame.dtype == np.uint8
    rgb_frame = rgb_frame.astype(np.float32) / 255.0
    assert rgb_frame.shape[2] == 3

    # Convert BGR to RGB
    rgb_frame = rgb_frame[:, :, ::-1]

    # Flatten RGB frame
    rgb_frame = rgb_frame.reshape(-1, 3)
    point_cloud = point_cloud.points

    # 6D point cloud
    point_cloud = np.concatenate((point_cloud, rgb_frame), axis=1)
    # Make float32
    point_cloud = point_cloud.astype(np.float32)

    # Save the point cloud
    save_folder.mkdir(parents=True, exist_ok=True)
    save_file = save_folder / f"colored_point_cloud_{idx:04d}.pkl"
    save_pickle(save_file, point_cloud)


def process_data_folder(data_folder: Path, gt_folder: Path, save_folder: Path) -> None:
    raw_depth_file = data_folder / "output.npz"
    raw_depth = load_npz(raw_depth_file)["depth"]
    # Frames, Height, Width

    print(f"Raw depth shape: {raw_depth.shape}")

    rgb_video_file = data_folder / "output_input.mp4"
    rgb_frames = load_frames_from_mp4(rgb_video_file)

    est_point_clouds = [
        orbbec_astra_disparity_to_point_cloud(raw_depth[frame])
        for frame in range(raw_depth.shape[0])
    ]
    gt_point_clouds = load_gt_point_clouds(gt_folder)

    assert len(est_point_clouds) == len(gt_point_clouds)
    assert len(est_point_clouds) == rgb_frames.shape[0]

    for idx, (rgb_frame, est_point_cloud, gt_point_cloud) in enumerate(
        zip(rgb_frames, est_point_clouds, gt_point_clouds)
    ):
        save_pkl_pointclouds(rgb_frame, est_point_cloud, save_folder, idx)

    visualizer = O3DVisualizer(add_world_frame=False)
    visualizer.add_world_frame()
    for point_cloud in est_point_clouds:
        visualizer.add_pointcloud(point_cloud, color=[1, 0, 0])
    for point_cloud in gt_point_clouds:
        visualizer.add_pointcloud(point_cloud, color=[0, 0, 1])
    visualizer.run()

    breakpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a folder containing data.")

    # Using Path directly as the type
    parser.add_argument("data_folder", type=Path, help="Path to the folder containing data")
    parser.add_argument(
        "gt_folder", type=Path, help="Path to the folder containing ground truth data"
    )
    parser.add_argument(
        "save_folder", type=Path, help="Path to the folder to save the processed data"
    )

    args = parser.parse_args()

    process_data_folder(args.data_folder, args.gt_folder, args.save_folder)


if __name__ == "__main__":
    main()
