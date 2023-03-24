import sys

sys.path.append('.')

import sys
import open3d as o3d
import numpy as np
import pypatchworkpp

from dataloaders import WaymoRawSequence, WaymoRawSequenceLoader
from pointclouds import PointCloud, SE3
import numpy as np
import tqdm
import argparse
from pathlib import Path
import collections
import open3d as o3d
import sklearn

from loader_utils import save_npy

# Get path to waymo data from command line
parser = argparse.ArgumentParser()
parser.add_argument('path', type=Path)
args = parser.parse_args()

params = pypatchworkpp.Parameters()
params.RNR_intensity_thr = 0.2
params.RNR_ver_angle_thr = -15.0
params.adaptive_seed_selection_margin = -1.2
params.elevation_thr = [0.0, 0.0, 0.0, 0.0]
params.enable_RNR = True
params.enable_RVPF = True
params.enable_TGR = True
params.flatness_thr = [0.0, 0.0, 0.0, 0.0]
params.intensity_thr = -8.728169361654678e-12
params.max_elevation_storage = 1000
params.max_flatness_storage = 1000
params.max_range = 80.0
params.min_range = 1
params.num_iter = 5
params.num_lpr = 20
params.num_min_pts = 10
params.num_rings_each_zone = [2, 4, 4, 4]
params.num_rings_of_interest = 4
params.num_sectors_each_zone = [16, 32, 54, 32]
params.num_zones = 4
params.sensor_height = 1.9304
params.th_dist = 0.125
params.th_dist_v = 0.1
params.th_seeds = 0.125
params.th_seeds_v = 0.25
params.uprightness_thr = 0.707
params.verbose = True

PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)


def process_sequence(sequence: WaymoRawSequence):
    for idx, entry_dict in enumerate(tqdm.tqdm(sequence.load_frame_list(None))):
        pc = entry_dict['relative_pc']

        pc_points = pc.points

        PatchworkPLUSPLUS.estimateGround(pc_points)
        # Does not provide ground mask, only ground points.
        ground_points = PatchworkPLUSPLUS.getGround()
        # Requires a second step to get the mask by doing NN matching.
        ground_point_idxes = sklearn.neighbors.NearestNeighbors(
            n_neighbors=1).fit(pc_points).kneighbors(
                ground_points, return_distance=False).squeeze(1)
        ground_mask = np.zeros(pc_points.shape[0], dtype=np.bool)
        ground_mask[ground_point_idxes] = True

        # Use open3d to visualize the pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        # Use the ground mask to color the points
        pcd.colors = o3d.utility.Vector3dVector(
            np.array([[0, 0, 1], [1, 0, 0]])[ground_mask.astype(np.int)])

        o3d.visualization.draw_geometries([pcd])

        save_path = args.path / 'ground_removed' / sequence.sequence_name / f'{idx:06d}.npy'
        save_npy(save_path, ground_point_idxes)


sequence_loader = WaymoRawSequenceLoader(args.path)
for sequence_id in sequence_loader.get_sequence_ids():
    process_sequence(sequence_loader.load_sequence(sequence_id))


def read_bin(bin_path):
    scan = np.fromfile(bin_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    return scan


# if __name__ == "__main__":

#     # Patchwork++ initialization
#     params = pypatchworkpp.Parameters()
#     params.verbose = True

#     PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

#     # Load point cloud
#     pointcloud = read_bin(input_cloud_filepath)

#     # Estimate Ground
#     PatchworkPLUSPLUS.estimateGround(pointcloud)

#     # Get Ground and Nonground
#     ground = PatchworkPLUSPLUS.getGround()
#     nonground = PatchworkPLUSPLUS.getNonground()
#     time_taken = PatchworkPLUSPLUS.getTimeTaken()

#     # Get centers and normals for patches
#     centers = PatchworkPLUSPLUS.getCenters()
#     normals = PatchworkPLUSPLUS.getNormals()

#     print("Origianl Points  #: ", pointcloud.shape[0])
#     print("Ground Points    #: ", ground.shape[0])
#     print("Nonground Points #: ", nonground.shape[0])
#     print("Time Taken : ", time_taken / 1000000, "(sec)")
#     print("Press ... \n")
#     print("\t H  : help")
#     print("\t N  : visualize the surface normals")
#     print("\tESC : close the Open3D window")

#     # Visualize
#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window(width=600, height=400)

#     mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

#     ground_o3d = o3d.geometry.PointCloud()
#     ground_o3d.points = o3d.utility.Vector3dVector(ground)
#     ground_o3d.colors = o3d.utility.Vector3dVector(
#         np.array([[0.0, 1.0, 0.0] for _ in range(ground.shape[0])],
#                  dtype=float)  # RGB
#     )

#     nonground_o3d = o3d.geometry.PointCloud()
#     nonground_o3d.points = o3d.utility.Vector3dVector(nonground)
#     nonground_o3d.colors = o3d.utility.Vector3dVector(
#         np.array([[1.0, 0.0, 0.0] for _ in range(nonground.shape[0])],
#                  dtype=float)  #RGB
#     )

#     centers_o3d = o3d.geometry.PointCloud()
#     centers_o3d.points = o3d.utility.Vector3dVector(centers)
#     centers_o3d.normals = o3d.utility.Vector3dVector(normals)
#     centers_o3d.colors = o3d.utility.Vector3dVector(
#         np.array([[1.0, 1.0, 0.0] for _ in range(centers.shape[0])],
#                  dtype=float)  #RGB
#     )

#     vis.add_geometry(mesh)
#     vis.add_geometry(ground_o3d)
#     vis.add_geometry(nonground_o3d)
#     vis.add_geometry(centers_o3d)

#     vis.run()
#     vis.destroy_window()