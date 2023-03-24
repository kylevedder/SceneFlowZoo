from typing import List

import numpy as np
import plotly.graph_objs as go
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps

import open3d as o3d


def polygon_to_points(polygon) -> np.ndarray:
    return [np.array([e.x, e.y, e.z]) for e in polygon]


def get_map_points(frame: dataset_pb2.Frame) -> np.ndarray:
    map_features = frame.map_features
    points = []
    for feature in map_features:
        if feature.HasField('road_edge'):
            points.extend(polygon_to_points(feature.road_edge.polyline))
        elif feature.HasField('crosswalk'):
            points.extend(polygon_to_points(feature.crosswalk.polygon))
        elif feature.HasField('road_line'):
            points.extend(polygon_to_points(feature.road_line.polyline))
    return np.array(points)


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
    # map_points = get_map_points(frame)

    # Use open3d to visualize the point cloud in xyz
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(car_frame_pc)

    # Use open3d to draw a 1 meter sphere at the origin
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)

    geometry_list = [pcd,  sphere]

    # for point in map_points:
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    #     sphere.translate(point)
    #     # Make the sphere red
    #     sphere.paint_uniform_color([1, 0, 0])
    #     geometry_list.append(sphere)

    o3d.visualization.draw_geometries(geometry_list)



import tensorflow as tf

FILENAME = '/efs/waymo_open_with_maps/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'

dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

# Load only 3 frames. Note that using too many frames may be slow to display.
frames = []
count = 0
for data in dataset:
    frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
    remove_ground(frame)
    # frames.append(frame)
    # count += 1
    # if count == 3:
    #     break

# Interactive plot of multiple point clouds aligned to the maps frame.

# For most systems:
#   left mouse button:   rotate
#   right mouse button:  pan
#   scroll wheel:        zoom

# plot_point_clouds_with_maps(frames)