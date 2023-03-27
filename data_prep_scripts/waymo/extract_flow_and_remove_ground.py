from sklearn.neighbors import NearestNeighbors
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from typing import Tuple, List, Dict
from pointclouds import PointCloud, SE3, SE2
from loader_utils import load_json, save_pickle

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import plot_maps

import open3d as o3d

GROUND_HEIGHT_THRESHOLD = 0.4  # 40 centimeters

parser = argparse.ArgumentParser()
parser.add_argument("flow_directory",
                    type=Path,
                    help="Path to Waymo flow data directory.")
parser.add_argument("heightmap_directory",
                    type=Path,
                    help="Path to rasterized heightmap.")
parser.add_argument("save_directory",
                    type=Path,
                    help="Path to save directory.")
parser.add_argument('--cpus',
                    type=int,
                    default=multiprocessing.cpu_count(),
                    help="Number of cpus to use for parallel processing")

args = parser.parse_args()


def parse_range_image_and_camera_projection(frame):
    """
    Parse range images and camera projections given a frame.

  Args:
     frame: open dataset frame proto

  Returns:
     range_images: A dict of {laser_name,
       [range_image_first_return, range_image_second_return]}.
     camera_projections: A dict of {laser_name,
       [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
  """
    range_images = {}
    camera_projections = {}
    point_flows = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if len(laser.ri_return1.range_image_flow_compressed) > 0:
                range_image_flow_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_flow_compressed, 'ZLIB')
                ri = dataset_pb2.MatrixFloat()
                ri.ParseFromString(
                    bytearray(range_image_flow_str_tensor.numpy()))
                point_flows[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytearray(range_image_top_pose_str_tensor.numpy()))

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]
        if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            if len(laser.ri_return2.range_image_flow_compressed) > 0:
                range_image_flow_str_tensor = tf.io.decode_compressed(
                    laser.ri_return2.range_image_flow_compressed, 'ZLIB')
                ri = dataset_pb2.MatrixFloat()
                ri.ParseFromString(
                    bytearray(range_image_flow_str_tensor.numpy()))
                point_flows[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)
    return range_images, camera_projections, point_flows, range_image_top_pose


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       point_flows,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False):
    """Convert range images to point cloud.

  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
      range_image_second_return]}.
    camera_projections: A dict of {laser_name,
      [camera_projection_from_first_return,
      camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.

  Returns:
    points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
      (NOTE: Will be {[N, 6]} if keep_polar_features is true.
    cp_points: {[N, 6]} list of camera projections of length 5
      (number of lidars).
  """
    calibrations = sorted(frame.context.laser_calibrations,
                          key=lambda c: c.name)
    points = []
    cp_points = []
    flows = []

    cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index,
        keep_polar_features)

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data),
            range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.compat.v1.where(range_image_mask))

        flow = point_flows[c.name][ri_index]
        flow_tensor = tf.reshape(tf.convert_to_tensor(value=flow.data),
                                 flow.shape.dims)
        flow_points_tensor = tf.gather_nd(flow_tensor,
                                          tf.compat.v1.where(range_image_mask))

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data),
                               cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor,
                                        tf.compat.v1.where(range_image_mask))

        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        flows.append(flow_points_tensor.numpy())

    return points, cp_points, flows


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


def get_car_pc_global_pc_flow_transform(
    frame: dataset_pb2.Frame
) -> Tuple[PointCloud, PointCloud, np.ndarray, SE3]:

    # Parse the frame lidar data into range images.
    range_images, camera_projections, point_flows, range_image_top_poses = parse_range_image_and_camera_projection(
        frame)

    # Project the range images into points.
    points_lst, cp_points, flows_lst = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        point_flows,
        range_image_top_poses,
        keep_polar_features=True)

    car_frame_pc = points_lst[0][:, 3:]
    car_frame_flows = flows_lst[0][:, :3]
    car_frame_labels = flows_lst[0][:, 3]
    num_points = car_frame_pc.shape[0]

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
    return PointCloud(car_frame_pc), PointCloud(
        world_frame_pc), car_frame_flows, car_frame_labels, SE3.from_array(
            car_to_global_transform)


def flow_path_to_height_map_path(flow_path: Path):
    heightmap_dir = args.heightmap_directory / flow_path.parent.name / (
        flow_path.stem + "_map")
    assert heightmap_dir.is_dir(), f"{heightmap_dir} is not a directory"
    return heightmap_dir


def flow_path_to_save_folder(flow_path: Path):
    save_folder = args.save_directory / flow_path.parent.name / flow_path.stem
    save_folder.mkdir(parents=True, exist_ok=True)
    return save_folder


def load_ground_height_raster(map_path: Path):
    raster_height_path = map_path / "ground_height.npy"
    transform_path = map_path / "se2.json"

    raster_heightmap = np.load(raster_height_path)
    transform = load_json(transform_path)

    transform_rotation = np.array(transform['R']).reshape(2, 2)
    transform_translation = np.array(transform['t'])
    transform_scale = np.array(transform['s'])

    transform_se2 = SE2(rotation=transform_rotation,
                        translation=transform_translation)

    return raster_heightmap, transform_se2, transform_scale


def get_ground_heights(raster_heightmap, global_to_raster_se2,
                       global_to_raster_scale,
                       global_point_cloud: PointCloud) -> np.ndarray:
    """Get ground height for each of the xy locations in a point cloud.
        Args:
            point_cloud: Numpy array of shape (k,2) or (k,3) in global coordinates.
        Returns:
            ground_height_values: Numpy array of shape (k,)
        """

    global_points_xy = global_point_cloud.points[:, :2]

    raster_points_xy = global_to_raster_se2.transform_point_cloud(
        global_points_xy) * global_to_raster_scale

    raster_points_xy = np.round(raster_points_xy).astype(np.int64)

    ground_height_values = np.full((raster_points_xy.shape[0]), np.nan)
    # outside max X
    outside_max_x = (raster_points_xy[:, 0] >=
                     raster_heightmap.shape[1]).astype(bool)
    # outside max Y
    outside_max_y = (raster_points_xy[:, 1] >=
                     raster_heightmap.shape[0]).astype(bool)
    # outside min X
    outside_min_x = (raster_points_xy[:, 0] < 0).astype(bool)
    # outside min Y
    outside_min_y = (raster_points_xy[:, 1] < 0).astype(bool)
    ind_valid_pts = ~np.logical_or(np.logical_or(outside_max_x, outside_max_y),
                                   np.logical_or(outside_min_x, outside_min_y))

    ground_height_values[ind_valid_pts] = raster_heightmap[raster_points_xy[
        ind_valid_pts, 1], raster_points_xy[ind_valid_pts, 0]]

    return ground_height_values


def is_ground_points(raster_heightmap, global_to_raster_se2,
                     global_to_raster_scale,
                     global_point_cloud: PointCloud) -> np.ndarray:
    """Remove ground points from a point cloud.
        Args:
            point_cloud: Numpy array of shape (k,3) in global coordinates.
        Returns:
            ground_removed_point_cloud: Numpy array of shape (k,3) in global coordinates.
        """
    ground_height_values = get_ground_heights(raster_heightmap,
                                              global_to_raster_se2,
                                              global_to_raster_scale,
                                              global_point_cloud)
    is_ground_boolean_arr = (
        np.absolute(global_point_cloud[:, 2] - ground_height_values) <=
        GROUND_HEIGHT_THRESHOLD) | (
            np.array(global_point_cloud[:, 2] - ground_height_values) < 0)
    return is_ground_boolean_arr


def visualize_point_cloud_flow(point_cloud: PointCloud, flow: np.ndarray):

    print("Visualizing point cloud and flow")
    # Use open3d to visualize the point cloud and flow.

    flowed_point_cloud = point_cloud.flow(flow[:, :3])

    geometries = []

    # Make base pointcloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    geometries.append(pcd)

    # Add line set
    line_set = o3d.geometry.LineSet()
    line_set_points = np.concatenate(
        [point_cloud.points, flowed_point_cloud.points], axis=0)

    lines = np.array([[i, i + len(point_cloud)]
                      for i in range(len(point_cloud))])
    line_set.points = o3d.utility.Vector3dVector(line_set_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # Line set is blue
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]
                                                  for _ in range(len(lines))])
    geometries.append(line_set)

    # Visualize the pointcloud
    o3d.visualization.draw_geometries(geometries)


def build_work_queue(waymo_directory):
    waymo_directory = Path(waymo_directory)
    assert waymo_directory.is_dir(), f"{waymo_directory} is not a directory"

    train_records = sorted((waymo_directory / 'training').glob('*.tfrecord'))
    val_records = sorted((waymo_directory / 'validation').glob('*.tfrecord'))

    queue = train_records + val_records
    for record in queue:
        assert record.is_file(), f"{record} is not a file"

    return queue


def process_record(file_path: Path):
    file_path = Path(file_path)
    print("Processing", file_path)
    height_map_path = flow_path_to_height_map_path(file_path)
    save_folder = flow_path_to_save_folder(file_path)
    raster_heightmap, transform_se2, transform_scale = load_ground_height_raster(
        height_map_path)

    print("LOADING FILEPATH", file_path)

    dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    for idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))

        car_frame_pc, global_frame_pc, flow, label, pose = get_car_pc_global_pc_flow_transform(
            frame)

        if (flow == -1).all():
            continue

        keep_points_mask = ~is_ground_points(raster_heightmap, transform_se2,
                                             transform_scale, global_frame_pc)

        masked_car_frame_pc = car_frame_pc.mask_points(keep_points_mask)
        masked_flow = flow[keep_points_mask] / 10.0
        masked_label = label[keep_points_mask]

        # visualize_point_cloud_flow(masked_car_frame_pc, masked_flow)

        save_pickle(
            save_folder / f"{idx:06d}.pkl", {
                "car_frame_pc": masked_car_frame_pc.points,
                "flow": masked_flow,
                "label": masked_label,
                "pose": pose.to_array(),
                "fraction_kept":
                np.sum(keep_points_mask) / len(keep_points_mask)
            })


work_queue = build_work_queue(args.flow_directory)
print("Work queue size:", len(work_queue))

assert args.cpus > 0, "Must have at least 1 CPU"
assert len(work_queue) > 0, "Work queue must have at least 1 element"

# for idx, record in enumerate(work_queue):
#     process_record(record)

num_cores = min(args.cpus, len(work_queue))
Parallel(num_cores)(delayed(process_record)(record) for record in work_queue)
