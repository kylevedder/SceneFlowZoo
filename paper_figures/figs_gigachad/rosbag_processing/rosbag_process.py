#!/usr/bin/env python3

import argparse
import rospy
import rosbag
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
from pathlib import Path
import pickle
import sensor_msgs.point_cloud2 as pc2
import tqdm
from typing import List


def get_topic_names(bag_path: str) -> List[str]:
    """
    Extract the names of topics from a rosbag file.

    Args:
        bag_path (str): Path to the rosbag file.

    Returns:
        List[str]: A list of topic names.
    """
    with rosbag.Bag(bag_path, "r") as bag:
        topics = bag.get_type_and_topic_info()[1].keys()
        return list(topics)


def point_cloud_callback(msg, output_folder):
    """
    Callback to save the PointCloud2 message as a pickle file.

    Args:
        msg (PointCloud2): The received point cloud message.
        output_folder (Path): The directory where the point clouds will be saved.
    """
    points = []
    for point in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
        x, y, z, rgb = point
        r = ((rgb >> 16) & 0x0000FF) / 255.0
        g = ((rgb >> 8) & 0x0000FF) / 255.0
        b = (rgb & 0x0000FF) / 255.0
        points.append((x, y, z, r, g, b))

    output_file = output_folder / f"colored_point_cloud_{rospy.Time.now()}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(points, f)
    print(f"Saved colored point cloud to {output_file}")


def publish_and_process(bag_path: Path, output_folder: Path):
    """
    Reads messages from a ROS bag, publishes them to the appropriate topics,
    and processes the resulting point clouds.

    Args:
        bag_path (Path): Path to the ROS bag file.
        output_folder (Path): Directory to save the output point clouds.
    """

    topic_names = get_topic_names(str(bag_path))
    print(f"Found topics: {topic_names}")

    rospy.init_node("rosbag_to_pointcloud", anonymous=True)

    bridge = CvBridge()

    # Publishers for camera info, color image, and registered depth image
    camera_info_pub = rospy.Publisher("rgb/camera_info", CameraInfo, queue_size=10)
    color_image_pub = rospy.Publisher("rgb/image_rect_color", Image, queue_size=10)
    depth_image_pub = rospy.Publisher("depth_registered/image_rect", Image, queue_size=10)

    output_folder.mkdir(parents=True, exist_ok=True)

    # Subscriber to get the point cloud output from the nodelet
    rospy.Subscriber("/depth_registered/points", PointCloud2, point_cloud_callback, output_folder)

    # Read from the bag and publish
    with rosbag.Bag(str(bag_path), "r") as bag:
        # Initialize iterators for each topic
        camera_info_msgs = list(
            bag.read_messages(
                topics=[
                    "/camera/color/camera_info",
                    "/camera/color/image_raw",
                    "/camera/depth/image_raw",
                ]
            )
        )
        for topic, msg, _ in tqdm.tqdm(camera_info_msgs):
            if topic == "/camera/color/camera_info":
                camera_info_pub.publish(msg)
            elif topic == "/camera/color/image_raw":
                color_image_pub.publish(msg)
            elif topic == "/camera/depth/image_raw":
                depth_image_pub.publish(msg)
            # Allow time for processing
            rospy.sleep(0.1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract colored point clouds from a ROS bag file using depth_image_proc."
    )
    parser.add_argument("bag_path", type=Path, help="Path to the ROS bag file")
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Output folder to save the colored point clouds",
    )
    args = parser.parse_args()

    publish_and_process(args.bag_path, args.output_folder)


if __name__ == "__main__":
    main()
