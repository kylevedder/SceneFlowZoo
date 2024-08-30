#!/usr/bin/env python3

import argparse
import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image
import cv2
from cv_bridge import CvBridge
from typing import List, Tuple, Any
from pathlib import Path
import tqdm
import pickle
import multiprocessing
from dataclasses import dataclass


@dataclass
class CameraInfo:
    K: np.ndarray
    R: np.ndarray
    P: np.ndarray


def process_image_pc(
    input_tuple: Tuple[np.ndarray, CameraInfo, Tuple[float, float, float]]
) -> np.ndarray:
    cv_image, camera_info, raw_point = input_tuple
    x, y, z = raw_point

    # Project 3D point to 2D image plane
    point_3d = np.array([x, y, z, 1.0]).reshape(4, 1)
    point_2d = camera_info.P @ point_3d

    # Normalize the 2D point
    point_2d = point_2d / point_2d[2]

    # Get pixel coordinates
    u, v = int(point_2d[0]), int(point_2d[1])

    # Check if the point is within the image boundaries
    height, width = cv_image.shape[:2]
    if 0 <= u < width and 0 <= v < height:
        # Get the color of the pixel
        b, g, r = cv_image[v, u] / 255.0
        return np.array([x, y, z, r, g, b], dtype=np.float32)
    else:
        # If the point is outside the image, return the 3D coordinates with black color
        return np.array([x, y, z, 0, 0, 0], dtype=np.float32)


def match_rgb_image_to_point_cloud(cloud_frames: list, image_frames: list) -> list:
    matched_msg_lst = []
    for _, pc_msg, cloud_t in cloud_frames:

        best_so_far = None
        best_so_far_diff = float("inf")
        # Iterate through image frames to find the closest match
        for _, image_msg, image_t in image_frames:
            diff = abs(cloud_t - image_t).to_sec()
            if diff < best_so_far_diff:
                best_so_far_diff = diff
                best_so_far = image_msg

        assert best_so_far is not None, "No matching image found for point cloud"
        matched_msg_lst.append((pc_msg, best_so_far))
    return matched_msg_lst


def extract_colored_point_clouds(
    bag_path: Path, cloud_topic: str, image_topic: str, color_camera_info_topic: str
) -> List[np.ndarray]:
    """
    Extract colored point clouds from a ROS bag file.

    Args:
        bag_path (Path): Path to the rosbag file.
        cloud_topic (str): Name of the point cloud topic.
        image_topic (str): Name of the image topic.

    Returns:
        List[np.ndarray]: A list of numpy arrays, each containing x, y, z, r, g, b columns as float32.
    """
    bridge = CvBridge()
    colored_point_clouds = []
    with multiprocessing.Pool() as pool:
        with rosbag.Bag(str(bag_path), "r") as bag:

            cloud_frames = list(bag.read_messages(topics=[cloud_topic]))
            image_frames = list(bag.read_messages(topics=[image_topic]))
            color_camera_info = list(bag.read_messages(topics=[color_camera_info_topic]))[0][1]
            K = np.array(color_camera_info.K).reshape(3, 3)
            R = np.array(color_camera_info.R).reshape(3, 3)
            P = np.array(color_camera_info.P).reshape(3, 4)
            color_camera_info = CameraInfo(K, R, P)

            matched_frames = match_rgb_image_to_point_cloud(cloud_frames, image_frames)

            for last_cloud_msg, last_image_msg in tqdm.tqdm(matched_frames):
                # Convert the image message to an OpenCV image
                cv_image = bridge.imgmsg_to_cv2(last_image_msg, desired_encoding="bgr8")
                ros_point_list = list(
                    pc2.read_points(last_cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
                )

                # Lists to hold point cloud data
                points_list = pool.map(
                    process_image_pc,
                    zip(
                        [cv_image] * len(ros_point_list),
                        [color_camera_info] * len(ros_point_list),
                        ros_point_list,
                    ),
                )
                point_cloud_array = np.array(points_list, dtype=np.float32)
                colored_point_clouds.append(point_cloud_array)

    return colored_point_clouds


def save_point_clouds_as_pickle(point_clouds: List[np.ndarray], output_folder: Path) -> None:
    """
    Save each colored point cloud as a separate pickle file.

    Args:
        point_clouds (List[np.ndarray]): List of numpy arrays with x, y, z, r, g, b columns as float32.
        output_folder (Path): The folder to save the pickle files.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, point_cloud in enumerate(point_clouds):
        output_file = output_folder / f"colored_point_cloud_{i:04d}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(point_cloud, f)
        print(f"Saved colored point cloud to {output_file}")


def main() -> None:
    """
    Main function to parse arguments and extract colored point clouds.
    """
    parser = argparse.ArgumentParser(
        description="Extract colored point clouds from a ROS bag file."
    )
    parser.add_argument("bag_path", type=Path, help="Path to the ROS bag file")
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Output folder to save the colored point clouds",
    )
    parser.add_argument(
        "--cloud_topic",
        type=str,
        default="/camera/depth/points",
        help="Name of the point cloud topic",
    )
    parser.add_argument(
        "--image_topic", type=str, default="/camera/color/image_raw", help="Name of the image topic"
    )
    parser.add_argument(
        "--color_camera_info_topic",
        type=str,
        default="/camera/color/camera_info",
        help="Name of the color camera info topic",
    )
    args = parser.parse_args()

    colored_point_clouds = extract_colored_point_clouds(
        args.bag_path, args.cloud_topic, args.image_topic, args.color_camera_info_topic
    )

    if colored_point_clouds:
        # Save each point cloud as a separate pickle file in the output folder
        save_point_clouds_as_pickle(colored_point_clouds, args.output_folder)
    else:
        print("No colored point clouds extracted.")


if __name__ == "__main__":
    main()
