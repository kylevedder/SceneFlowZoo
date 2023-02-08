import torch
import pandas as pd
import open3d as o3d
from dataloaders import ArgoverseSequenceLoader, ArgoverseFlowSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import chamfer_distance

# sequence_loader = ArgoverseSequenceLoader('/bigdata/argoverse_lidar/train/')
sequence_loader = ArgoverseFlowSequenceLoader(
    '/efs/argoverse2/val/', '/efs/argoverse2/val_sceneflow/')

sequence_id = sequence_loader.get_sequence_ids()[29]
print("Sequence ID: ", sequence_id)
sequence = sequence_loader.load_sequence(sequence_id)

sequence_length = len(sequence)

frame_list = sequence.load_frame_list(0)


def sequence_idx_to_color(idx):
    return [1 - idx / sequence_length, idx / sequence_length, 0]


MAX_K = 10
current_k = 1
geometry_idx = 0
knn_result_cache = {}


def next_geometry(_):
    global geometry_idx
    geometry_idx = (geometry_idx + 1) % len(frame_list)
    draw_geometry(geometry_idx)


def previous_geometry(_):
    global geometry_idx
    geometry_idx = (geometry_idx - 1) % len(frame_list)
    draw_geometry(geometry_idx)


def cycle_k(_):
    global current_k
    current_k = (current_k + 1) % (MAX_K + 1)
    if current_k == 0:
        current_k = 1
    print("K: ", current_k)
    draw_geometry(geometry_idx)


def pc_knn(warped_pc: torch.Tensor,
           target_pc: torch.Tensor,
           K: int = 1) -> np.ndarray:
    if warped_pc.ndim == 2:
        warped_pc = warped_pc.unsqueeze(0)
    if target_pc.ndim == 2:
        target_pc = target_pc.unsqueeze(0)

    # Compute min distance between warped point cloud and point cloud at t+1.
    print("Computing knn...")
    warped_to_target_knn = knn_points(p1=warped_pc,
                                      p2=target_pc,
                                      K=K,
                                      return_nn=True)
    print("Done computing knn.")

    nearest_points = warped_to_target_knn.knn.squeeze(0)
    return nearest_points.cpu().numpy()


# make open3d visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window("Point Cloud Nearest Neighbor Visualization")
vis.get_render_option().point_size = 1.5
vis.get_render_option().background_color = (0, 0, 0)
vis.get_render_option().show_coordinate_frame = True
vis.register_key_callback(ord('K'), cycle_k)
# set up vector
vis.get_view_control().set_up([0, 0, 1])


def draw_geometry(idx: int, reset_boxes=False):
    frame_dict = frame_list[idx]
    pc = frame_dict['relative_pc']
    pose = frame_dict['relative_pose']
    flowed_pc = frame_dict['relative_flowed_pc']
    classes = frame_dict['pc_classes']
    is_grounds = frame_dict['pc_is_ground']

    vis.clear_geometries()

    # Add center of mass
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    sphere.translate(pose.translation)
    sphere.paint_uniform_color(sequence_idx_to_color(idx))
    vis.add_geometry(sphere, reset_bounding_box=reset_boxes)

    # Add base point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.points)
    pc_color = np.zeros_like(pc.points)
    # Base point cloud is red
    pc_color[:, 0] = 1.0
    pcd.colors = o3d.utility.Vector3dVector(pc_color)
    vis.add_geometry(pcd, reset_bounding_box=reset_boxes)

    if flowed_pc is None:
        return

    # Add gt pc from next frame
    next_pc = frame_list[idx + 1]['relative_pc']
    next_pcd = o3d.geometry.PointCloud()
    next_pcd.points = o3d.utility.Vector3dVector(next_pc.points)
    next_pc_color = np.zeros_like(next_pc.points)
    # Next point cloud is green
    next_pc_color[:, 1] = 1.0
    next_pcd.colors = o3d.utility.Vector3dVector(next_pc_color)
    vis.add_geometry(next_pcd, reset_bounding_box=reset_boxes)

    # Add line set between pc0 and pc1 using KNN
    pc_torch = torch.from_numpy(pc.points).float()
    next_pc_torch = torch.from_numpy(next_pc.points).float()

    if idx not in knn_result_cache:
        knn_result_cache[idx] = pc_knn(pc_torch, next_pc_torch, K=MAX_K)
    nearest_points_arr = knn_result_cache[idx]

    def get_k_nearest_points(nearest_points_arr, k):
        return nearest_points_arr[:, :k, :].mean(axis=1)

    nearest_points = get_k_nearest_points(nearest_points_arr, k=current_k)

    assert pc.shape == nearest_points.shape, f'pc and nearest_points must have same shape, but got {pc.shape} and {nearest_points.shape}'

    line_set = o3d.geometry.LineSet()
    line_set_points = np.concatenate([pc, nearest_points], axis=0)

    lines = np.array([[i, i + len(pc)] for i in range(len(pc))])
    line_set.points = o3d.utility.Vector3dVector(line_set_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # Line set is blue
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 1]
                                                  for _ in range(len(lines))])
    vis.add_geometry(line_set, reset_bounding_box=reset_boxes)


draw_geometry(geometry_idx, reset_boxes=True)

vis.run()
vis.destroy_window()