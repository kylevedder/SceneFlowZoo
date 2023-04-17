# Source code adapted from:
# https://github.com/waymo-research/waymo-open-dataset/blob/bbcd77fc503622a292f0928bfa455f190ca5946e/waymo_open_dataset/utils/frame_utils.py

import glob
import os
import pickle
import re

import cv2
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2, dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import tqdm


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
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    flows = []

    cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.compat.v1.where(range_image_mask))

        flow = point_flows[c.name][ri_index]
        flow_tensor = tf.reshape(tf.convert_to_tensor(value=flow.data), flow.shape.dims)
        flow_points_tensor = tf.gather_nd(flow_tensor,
                                          tf.compat.v1.where(range_image_mask))

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor,
                                        tf.compat.v1.where(range_image_mask))

        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        flows.append(flow_points_tensor.numpy())

    return points, cp_points, flows


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
                ri.ParseFromString(bytearray(range_image_flow_str_tensor.numpy()))
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
                ri.ParseFromString(bytearray(range_image_flow_str_tensor.numpy()))
                point_flows[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)
    return range_images, camera_projections, point_flows, range_image_top_pose


def save_point_cloud(compressed_frame, file_path):
    """
    Compute the point cloud from a frame and stores it into disk.
    :param compressed_frame: compressed frame from a TFRecord
    :param file_path: name path that will have the stored point cloud
    :returns:
        - points - [N, 5] matrix which stores the [x, y, z, intensity, elongation] in the frame reference
        - flows - [N, 4] matrix where each row is the flow for each point in the form [vx, vy, vz, label]
                  in the reference frame
        - transform - [,16] flattened transformation matrix
    """
    frame = get_uncompressed_frame(compressed_frame)
    points, flows = compute_features(frame)
    point_cloud = np.hstack((points, flows))
    np.savez_compressed(file_path, frame=point_cloud)
    transform = list(frame.pose.transform)
    return points, flows, transform


def preprocess(tfrecord_file, output_path, frames_per_segment = None):
    """
    TFRecord file to store in a suitable form for training
    in disk. A point cloud in disk has dimensions [N, 9] where N is the number of points
    and per each point it stores [x, y, z, intensity, elongation, vx, vy, vz, label].
    It stores in a dictionary relevant metadata:
        - look-up table: It has the form [[t_1, t_0], [t_2, t_1], ... , [t_n, t_(n-1)]], where t_i is
        (file_path, transform), where file_path is the file where the point cloud is stored and transform the transformation
        to apply to a point to change it reference frame from global to the car frame in that moment.
        - flows information: min and max flows encountered for then visualize pointclouds properly

    :param tfrecord_file: TFRecord file. It should have the flow extension.
                          They can be downloaded from https://console.cloud.google.com/storage/browser/waymo_open_dataset_scene_flow
    :param output_path: path where the processed point clouds will be saved.
    """
    tfrecord_filename = os.path.basename(tfrecord_file)
    tfrecord_filename = os.path.splitext(tfrecord_filename)[0]

    look_up_table = []
    metadata_path = os.path.join(output_path, f"metadata_{tfrecord_filename}")  # look-up table and flows mins and maxs
    loaded_file = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
    previous_frame = None

    # Needed of max and min flows for normalizing in visualization
    min_vx_global, max_vx_global = np.inf, -np.inf
    min_vy_global, max_vy_global = np.inf, -np.inf
    min_vz_global, max_vz_global = np.inf, -np.inf

    for j, frame in enumerate(tqdm.tqdm(loaded_file)):
        output_file_name = f"pointCloud_file_{tfrecord_filename}_frame_{j:06d}.npz"
        point_cloud_path = os.path.join(output_path, output_file_name)
        # Process frame and store point clouds into disk
        _, flows, pose_transform = save_point_cloud(frame, point_cloud_path)
        flows = flows[flows[:, -1] != -1] 
        if flows.size != 0:  # May all the flows are invalid
            min_vx, min_vy, min_vz = flows[:, :-1].min(axis=0)
            max_vx, max_vy, max_vz = flows[:, :-1].max(axis=0)
            min_vx_global = min(min_vx_global, min_vx)
            min_vy_global = min(min_vy_global, min_vy)
            min_vz_global = min(min_vz_global, min_vz)
            max_vx_global = max(max_vx_global, max_vx)
            max_vy_global = max(max_vy_global, max_vy)
            max_vz_global = max(max_vz_global, max_vz)

        if j == 0:
            previous_frame = (output_file_name, pose_transform)
        else:
            current_frame = (output_file_name, pose_transform)
            look_up_table.append([current_frame, previous_frame])
            previous_frame = current_frame
        if frames_per_segment is not None and j == frames_per_segment:
            break

    # Save metadata into disk
    flows_info = {'min_vx': min_vx_global,
                  'max_vx': max_vx_global,
                  'min_vy': min_vy_global,
                  'max_vy': max_vy_global,
                  'min_vz': min_vz_global,
                  'max_vz': max_vz_global}

    metadata = {'look_up_table': look_up_table,
                'flows_information': flows_info}

    with open(metadata_path, 'wb') as metadata_file:
        pickle.dump(metadata, metadata_file)


def get_uncompressed_frame(compressed_frame):
    """
    :param compressed_frame: Compressed frame
    :return: Uncompressed frame
    """
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(compressed_frame.numpy()))
    return frame


def compute_features(frame):
    """
    :param frame: Uncompressed frame
    :return: [N, F], [N, 4], where N is the number of points, F the number of features,
    which is [x, y, z, intensity, elongation] and 4 in the second results stands for [vx, vy, vz, label],
    which corresponds to the flow information
    """
    range_images, camera_projections, point_flows, range_image_top_pose = parse_range_image_and_camera_projection(
        frame)

    points, cp_points, flows = convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        point_flows,
        range_image_top_pose,
        keep_polar_features=True)

    # 3D points in the vehicle reference frame
    points_all = np.concatenate(points, axis=0)
    flows_all = np.concatenate(flows, axis=0)
    # We skip the range feature since pillars will account for it
    # Note that first are features and then point coordinates
    points_features, points_coord = points_all[:, 1:3], points_all[:, 3:points_all.shape[1]]
    points_all = np.hstack((points_coord, points_features))
    return points_all, flows_all


def merge_metadata(input_path):
    """
    Merge individual look-up table and flows mins and maxs and store it in the input_path with the name metadata
    :param input_path: Path with the local metadata in the form metadata_[tfRecordName]
    """
    look_up_table = []
    flows_info = None

    os.chdir(input_path)
    for file in glob.glob("metadata_*"):
        file_name = os.path.abspath(file)
        try:
            with open(file_name, 'rb') as metadata_file:
                metadata_local = pickle.load(metadata_file)
                look_up_table.extend(metadata_local['look_up_table'])
                flows_information = metadata_local['flows_information']
                if flows_info is None:
                    flows_info = flows_information
                else:
                    flows_info['min_vx'] = min(flows_info['min_vx'], flows_information['min_vx'])
                    flows_info['min_vx'] = min(flows_info['min_vx'], flows_information['min_vy'])
                    flows_info['min_vz'] = min(flows_info['min_vz'], flows_information['min_vz'])
                    flows_info['max_vx'] = max(flows_info['max_vx'], flows_information['max_vx'])
                    flows_info['max_vy'] = max(flows_info['max_vy'], flows_information['max_vy'])
                    flows_info['max_vz'] = max(flows_info['max_vz'], flows_information['max_vz'])

        except FileNotFoundError:
            raise FileNotFoundError(
                "Metadata not found when merging individual metadata")

    # Save metadata into disk
    metadata = {'look_up_table': look_up_table,
                'flows_information': flows_info}
    with open(os.path.join(input_path, "metadata"), 'wb') as metadata_file:
        pickle.dump(metadata, metadata_file)


def load_pfm(filename):
    file = open(filename, 'r', newline='', encoding='latin-1')

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    file.close()

    return np.reshape(data, shape), scale


def bilinear_interp_val(vmap, y, x):
    """
    bilinear interpolation on a 2D map
    """
    h, w = vmap.shape
    x1 = int(x)
    x2 = x1 + 1
    x2 = w - 1 if x2 > (w - 1) else x2
    y1 = int(y)
    y2 = y1 + 1
    y2 = h - 1 if y2 > (h - 1) else y2
    Q11 = vmap[y1, x1]
    Q21 = vmap[y1, x2]
    Q12 = vmap[y2, x1]
    Q22 = vmap[y2, x2]
    return Q11 * (x2 - x) * (y2 - y) + Q21 * (x - x1) * (y2 - y) + Q12 * (x2 - x) * (y - y1) + Q22 * (x - x1) * (y - y1)


def get_3d_pos_xy(y_prime, x_prime, depth, focal_length=1050., w=960, h=540):
    """ depth pop up """
    y = (y_prime - h / 2.) * depth / focal_length
    x = (x_prime - w / 2.) * depth / focal_length
    return [x, y, depth]


def readFlow(name):
    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def generate_flying_things_point_cloud(fname_disparity, fname_disparity_next_frame, fname_disparity_change,
                                       fname_optical_flow, image, image_next_frame, max_cut=35, focal_length=1050.,
                                       n = 2048, add_label=True):
    # generate needed data
    disparity_np, _ = load_pfm(fname_disparity)
    disparity_next_frame_np, _ = load_pfm(fname_disparity_next_frame)
    disparity_change_np, _ = load_pfm(fname_disparity_change)
    optical_flow_np = readFlow(fname_optical_flow)
    rgb_np = cv2.imread(image)[:, :, ::-1] / 255.
    rgb_next_frame_np = cv2.imread(image_next_frame)[:, :, ::-1] / 255.

    depth_np = focal_length / disparity_np
    depth_next_frame_np = focal_length / disparity_next_frame_np
    future_depth_np = focal_length / (disparity_np + disparity_change_np)
    h, w = disparity_np.shape

    try:
        depth_requirement = depth_np < max_cut
    except:
        return None

    satisfy_pix1 = np.column_stack(np.where(depth_requirement))
    if satisfy_pix1.shape[0] < n:
        return None
    sample_choice1 = np.random.choice(satisfy_pix1.shape[0], size=n, replace=False)
    sampled_pix1_x = satisfy_pix1[sample_choice1, 1]
    sampled_pix1_y = satisfy_pix1[sample_choice1, 0]
    n_1 = sampled_pix1_x.shape[0]
    current_pos1 = np.array([get_3d_pos_xy(sampled_pix1_y[i], sampled_pix1_x[i],
                                           depth_np[int(sampled_pix1_y[i]),
                                                    int(sampled_pix1_x[i])]) for i in range(n_1)])

    current_rgb1 = np.array([[rgb_np[h - 1 - int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 0],
                              rgb_np[h - 1 - int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 1],
                              rgb_np[h - 1 - int(sampled_pix1_y[i]), int(sampled_pix1_x[i]), 2]] for i in range(n)])

    # sampled_optical_flow_x = np.array(
    #     [optical_flow_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])][0] for i in range(n_1)])
    # sampled_optical_flow_y = np.array(
    #     [optical_flow_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])][1] for i in range(n_1)])
    # future_pix1_x = sampled_pix1_x + sampled_optical_flow_x
    # future_pix1_y = sampled_pix1_y - sampled_optical_flow_y
    # future_pos1 = np.array([get_3d_pos_xy(future_pix1_y[i], future_pix1_x[i],
    #                                       future_depth_np[int(sampled_pix1_y[i]), int(sampled_pix1_x[i])]) for i in
    #                         range(n_1)])
    # flow = future_pos1 - current_pos1

    try:
        depth_requirement = depth_next_frame_np < max_cut
    except:
        return None

    satisfy_pix2 = np.column_stack(np.where(depth_next_frame_np < max_cut))
    if satisfy_pix2.shape[0] < n:
        return None
    sample_choice2 = np.random.choice(satisfy_pix2.shape[0], size=n, replace=False)
    sampled_pix2_x = satisfy_pix2[sample_choice2, 1]
    sampled_pix2_y = satisfy_pix2[sample_choice2, 0]
    n_2 = sampled_pix2_x.shape[0]

    current_pos2 = np.array([get_3d_pos_xy(sampled_pix2_y[i], sampled_pix2_x[i],
                                           depth_next_frame_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])]) for i in
                             range(n_2)])

    current_rgb2 = np.array([[rgb_next_frame_np[h-1-int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 0],
                              rgb_next_frame_np[h-1-int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 1],
                              rgb_next_frame_np[h-1-int(sampled_pix2_y[i]), int(sampled_pix2_x[i]), 2]]
                             for i in range(n)])

    # Compute backward flow
    sampled_optical_flow_x = np.array(
        [optical_flow_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])][0] for i in range(n_2)])
    sampled_optical_flow_y = np.array(
        [optical_flow_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])][1] for i in range(n_2)])
    future_pix2_x = sampled_pix2_x + sampled_optical_flow_x
    future_pix2_y = sampled_pix2_y - sampled_optical_flow_y
    future_pos2 = np.array([get_3d_pos_xy(future_pix2_y[i], future_pix2_x[i],
                                          future_depth_np[int(sampled_pix2_y[i]), int(sampled_pix2_x[i])]) for i in
                            range(n_2)])
    flow = future_pos2 - current_pos2

    # mask, judge whether point move out of fov or occluded by other object after motion
    future_pos2_depth = future_depth_np[sampled_pix2_y, sampled_pix2_x]
    future_pos2_foreground_depth = np.zeros_like(future_pos2_depth)
    valid_mask_fov2 = np.ones_like(future_pos2_depth, dtype=bool)
    for i in range(future_pos2_depth.shape[0]):
        if 0 < future_pix2_y[i] < h and 0 < future_pix2_x[i] < w:
            future_pos2_foreground_depth[i] = bilinear_interp_val(depth_next_frame_np, future_pix2_y[i], future_pix2_x[i])
        else:
            valid_mask_fov2[i] = False
    valid_mask_occ2 = (future_pos2_foreground_depth - future_pos2_depth) > -5e-1

    mask2 = valid_mask_occ2 & valid_mask_fov2

    mask2 = np.ones(shape=mask2.shape)

    # Add redundant zero labels for each flow in order to fit the shape
    if add_label:
        labelled_flow = np.ones(shape=(flow.shape[0], flow.shape[1] + 1))
        labelled_flow[:, :-1] = flow
        flow = labelled_flow

    return current_pos1, current_pos2, current_rgb1, current_rgb2, flow, mask2


def get_all_flying_things_frames(input_dir, disp_dir, opt_dir, disp_change_dir, img_dir):
    all_files_disparity = glob.glob(os.path.join(input_dir, '{0}/*.pfm'.format(disp_dir)))
    all_files_disparity_change = glob.glob(os.path.join(input_dir, '{0}/*.pfm'.format(disp_change_dir)))
    all_files_opt_flow = glob.glob(os.path.join(input_dir, '{0}/*.flo'.format(opt_dir)))
    all_files_img = glob.glob(os.path.join(input_dir, '{0}/*.png'.format(img_dir)))

    return all_files_disparity, all_files_disparity_change, all_files_opt_flow, all_files_img