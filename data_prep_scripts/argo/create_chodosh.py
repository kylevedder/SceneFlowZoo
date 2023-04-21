import os

# Set Open MP's num threads to 1
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import numpy as np
import multiprocessing
from multiprocessing import Pool, cpu_count
from sklearn.cluster import DBSCAN
import time
from tqdm import tqdm

# Take in input and output folder with no default arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_folder", help="Path to the input folder")
parser.add_argument("output_folder", help="Path to the output folder")
parser.add_argument("--num_cpus",
                    type=int,
                    default=multiprocessing.cpu_count(),
                    help="Number of CPUs to use for parallel processing")

# Parse the arguments
args = parser.parse_args()

# Verify that the input folder is a folder and exists
input_folder_path = Path(args.input_folder)
assert input_folder_path.exists(), f"Error: {input_folder_path} does not exist"
assert input_folder_path.is_dir(
), f"Error: {input_folder_path} is not a directory"

# Create the output folder if it doesn't exist
output_folder_path = Path(args.output_folder)
output_folder_path.mkdir(parents=True, exist_ok=True)

# Load all the .npz file paths in the input folder into a list
npz_file_list = list(input_folder_path.glob("*.npz"))

#########################################################


def fit_rigid(A, B):
    """
    Fit Rigid transformation A @ R.T + t = B
    """
    assert A.shape == B.shape
    num_rows, num_cols = A.shape

    # find mean column wise
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # ensure centroids are 1x3
    centroid_A = centroid_A.reshape(1, -1)
    centroid_B = centroid_B.reshape(1, -1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -centroid_A @ R.T + centroid_B

    return R, t


def ransac_fit_rigid(pts1, pts2, inlier_thresh, ntrials):
    best_R = np.eye(3)
    best_t = np.zeros((3, ))
    best_inliers = (np.linalg.norm(pts1 - pts2, axis=-1) < inlier_thresh)
    best_inliers_sum = best_inliers.sum()
    for i in range(ntrials):
        choice = np.random.choice(len(pts1), 3)
        R, t = fit_rigid(pts1[choice], pts2[choice])
        inliers = (np.linalg.norm(pts1 @ R.T + t - pts2, axis=-1) <
                   inlier_thresh)
        if inliers.sum() > best_inliers_sum:
            best_R = R
            best_t = t
            best_inliers = inliers
            best_inliers_sum = best_inliers.sum()
            if best_inliers_sum / len(pts1) > 0.5:
                break
    best_R, best_t = fit_rigid(pts1[best_inliers], pts2[best_inliers])
    return best_R, best_t


def refine_flow(pc0,
                flow_pred,
                eps=0.4,
                min_points=10,
                motion_threshold=0.05,
                inlier_thresh=0.2,
                ntrials=250):
    labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(pc0)
    max_label = labels.max()
    refined_flow = np.zeros_like(flow_pred)
    for l in range(max_label + 1):
        label_mask = labels == l
        cluster_pts = pc0[label_mask]
        cluster_flows = flow_pred[label_mask]
        R, t = ransac_fit_rigid(cluster_pts, cluster_pts + cluster_flows,
                                inlier_thresh, ntrials)
        if np.linalg.norm(t) < motion_threshold:
            R = np.eye(3)
            t = np.zeros((3, ))
        refined_flow[label_mask] = (cluster_pts @ R.T + t) - cluster_pts

    refined_flow[labels == -1] = flow_pred[labels == -1]
    return refined_flow


#########################################################


# Define the worker function for parallel processing
def process_npz_file(npz_path):
    # Load the data from the .npz file
    data = dict(np.load(npz_path))
    before_time = time.time()
    refine_flow(data["pc"], data["flow"])
    after_time = time.time()
    delta_time = after_time - before_time
    data["refine_delta_time"] = delta_time
    data["delta_time"] = delta_time + data["nsfp_delta_time"]
    npz_name = npz_path.stem
    folder_name, index_name, minibatch_name = npz_name.split("_")
    # Convert index_name to int
    index_name = int(index_name)
    # Convert minibatch_name to int
    minibatch_name = int(minibatch_name)
    # Convert 
    output_file_path = output_folder_path / folder_name / f"{index_name:010d}_{minibatch_name:03d}.npz"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file_path, **data)


# Process the .npz files either in parallel or with a standard for loop
if args.num_cpus <= 1:
    for npz_path in tqdm(npz_file_list):
        process_npz_file(npz_path)
else:
    print(f"Using {args.num_cpus} CPUs for parallel processing...")
    with Pool(processes=args.num_cpus) as pool:
        for _ in tqdm(pool.imap_unordered(process_npz_file, npz_file_list),
                      total=len(npz_file_list)):
            pass
