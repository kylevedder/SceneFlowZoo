from sklearn.neighbors import NearestNeighbors
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from typing import List

import numpy as np
import plotly.graph_objs as go
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps

import tensorflow as tf

import open3d as o3d

from pointclouds import PointCloud as myPC, SE3 as mySE3, SE2 as mySE2

# from sklearn.neighbors import NearestNeighbors
# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
# import argparse
# import multiprocessing
# from pathlib import Path
# from joblib import Parallel, delayed
# import numpy as np
# from typing import Tuple, List, Dict
# from pointclouds import PointCloud, SE3, SE2
# from loader_utils import load_json, save_npy

# from waymo_open_dataset import dataset_pb2
# from waymo_open_dataset.utils import frame_utils
# from waymo_open_dataset.utils import plot_maps

# import open3d as o3d



def get_pc(frame: dataset_pb2.Frame) -> np.ndarray:
    # Parse the frame lidar data into range images.
    range_images, camera_projections, seg_labels, range_image_top_poses = (
        frame_utils.parse_range_image_and_camera_projection(frame))
    # Project the range images into points.
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_poses,
        keep_polar_features=True,
    )
    car_frame_pc = points[0][:, 3:]
    num_points = car_frame_pc.shape[0]
    print(f'num_points: {num_points}')

    # Transform the points from the vehicle frame to the world frame.
    world_frame_pc = np.concatenate(
        [car_frame_pc, np.ones([num_points, 1])], axis=-1)
    car_to_global_transform = np.reshape(np.array(frame.pose.transform),
                                         [4, 4])
    world_frame_pc = np.transpose(
        np.matmul(car_to_global_transform, np.transpose(world_frame_pc)))[:,
                                                                          0:3]

    # Transform the points from the world frame to the map frame.
    offset = frame.map_pose_offset
    points_offset = np.array([offset.x, offset.y, offset.z])
    world_frame_pc += points_offset

    return car_frame_pc, world_frame_pc


def remove_ground(frame: dataset_pb2.Frame) -> None:
    car_frame_pc, global_frame_pc = get_pc(frame)

    # Use open3d to visualize the point cloud in xyz
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(car_frame_pc)

    # Use open3d to draw a 1 meter sphere at the origin
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    geometry_list = [pcd, sphere]

    o3d.visualization.draw_geometries(geometry_list)


FILENAME = '/efs/waymo_open_with_maps/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'

dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

for data in dataset:
    frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
    remove_ground(frame)



# def get_pc(frame: dataset_pb2.Frame) -> np.ndarray:
#     # Parse the frame lidar data into range images.
#     range_images, camera_projections, seg_labels, range_image_top_poses = (
#         frame_utils.parse_range_image_and_camera_projection(frame))
#     # Project the range images into points.
#     points, cp_points = frame_utils.convert_range_image_to_point_cloud(
#         frame,
#         range_images,
#         camera_projections,
#         range_image_top_poses,
#         keep_polar_features=True,
#     )
#     car_frame_pc = points[0][:, 3:]
#     num_points = car_frame_pc.shape[0]
#     print(f'num_points: {num_points}')

#     # Transform the points from the vehicle frame to the world frame.
#     world_frame_pc = np.concatenate(
#         [car_frame_pc, np.ones([num_points, 1])], axis=-1)
#     car_to_global_transform = np.reshape(np.array(frame.pose.transform),
#                                          [4, 4])
#     world_frame_pc = np.transpose(
#         np.matmul(car_to_global_transform, np.transpose(world_frame_pc)))[:,
#                                                                           0:3]

#     # Transform the points from the world frame to the map frame.
#     offset = frame.map_pose_offset
#     points_offset = np.array([offset.x, offset.y, offset.z])
#     world_frame_pc += points_offset

#     return car_frame_pc, world_frame_pc




# def remove_ground(frame: dataset_pb2.Frame) -> None:
#     car_frame_pc, global_frame_pc = get_pc(frame)

#     # Use open3d to visualize the point cloud in xyz
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(car_frame_pc)

#     # Use open3d to draw a 1 meter sphere at the origin
#     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

#     geometry_list = [pcd, sphere]

#     o3d.visualization.draw_geometries(geometry_list)

   

# FILENAME = '/efs/waymo_open_with_maps/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'
# print("LOADING FILEPATH", FILENAME)

# dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
# for data in dataset:
#     frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
#     remove_ground(frame)

