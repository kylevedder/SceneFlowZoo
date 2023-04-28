import argparse
import multiprocessing

from pathlib import Path
from loader_utils import load_npy, load_txt, save_txt
import numpy as np

import open3d as o3d
import tqdm

# Take as arguments the two folders to compare

parser = argparse.ArgumentParser(description='Process some folders.')
parser.add_argument('folder1', type=Path, help='First folder to compare')
parser.add_argument('folder2', type=Path, help='Second folder to compare')
parser.add_argument('--cpus',
                    type=int,
                    default=multiprocessing.cpu_count(),
                    help='Number of CPUs to use for multiprocessing')
args = parser.parse_args()

folder1 = args.folder1
folder2 = args.folder2

assert folder1.exists(), f"Error: {folder1} does not exist"
assert folder2.exists(), f"Error: {folder2} does not exist"

SIGNFICANT_DISAGREEMENT_THRESHOLD = 0.5
NUM_DISAGREEMENTS = 50


def flows_disagree(flow1, flow2):
    assert flow1.shape == flow2.shape, f"Error: {flow1} and {flow2} do not have the same shape ({flow1.shape} vs {flow2.shape})"
    distances = np.linalg.norm(flow1 - flow2, axis=1)
    num_disagreements = np.sum(distances > SIGNFICANT_DISAGREEMENT_THRESHOLD)
    return num_disagreements > NUM_DISAGREEMENTS


def make_lineset(pc1, pc2, color=[1, 0, 0]):
    line_set = o3d.geometry.LineSet()
    line_set_points = np.concatenate([pc1, pc2], axis=0)

    lines = np.array([[i, i + len(pc1)] for i in range(len(pc1))])
    line_set.points = o3d.utility.Vector3dVector(line_set_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # Line set is blue
    line_set.colors = o3d.utility.Vector3dVector(
        [color for _ in range(len(lines))])
    return line_set


def visualize_pc_and_flows(pc1, pc2, flow_gt, flow1, flow2):
    """
    Visualize the point cloud with flow1 in red and flow2 in blue.

    Use Open3d to draw the point cloud, the two flowed PCs, and line sets between the original PC and the two flowed PCs.
    """

    status = {
        "skip": False,
        "save": False,
    }

    def blacklist_callback(vis):
        status["skip"] = True
        vis.close()

    def whitelist_callback(vis):
        status["save"] = True
        # Save screenshot of the visualization
        screenshot_path = Path(
        ) / "anomaly_screenshots" / f"{file1.parent.name}" / f"{file1.stem}.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(screenshot_path))
        vis.close()

    flowed_pc1 = pc1 + flow1
    flowed_pc2 = pc1 + flow2
    flowed_gt = pc1 + flow_gt

    # Create Open3D point cloud object from the input pc1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc1)
    pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0]
                                             for _ in range(len(pc1))])

    # Create Open3D point cloud object from the input pc2
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd2.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]
                                              for _ in range(len(pc2))])

    # Create Open3D point cloud objects for the flowed point clouds
    pcd_flowed1 = o3d.geometry.PointCloud()
    pcd_flowed1.points = o3d.utility.Vector3dVector(flowed_pc1)

    pcd_flowed2 = o3d.geometry.PointCloud()
    pcd_flowed2.points = o3d.utility.Vector3dVector(flowed_pc2)

    # Create Open3D line sets between the original PC and the two flowed PCs
    lineset_flow1 = make_lineset(pc1, flowed_pc1, color=[1, 0, 0])
    lineset_flow2 = make_lineset(pc1, flowed_pc2, color=[0, 0, 1])
    lineset_flow_gt = make_lineset(pc1, flowed_gt, color=[0, 1, 0])
    # Create an Open3D visualization window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord('N'), blacklist_callback)
    vis.register_key_callback(ord('Y'), whitelist_callback)
    # vis.register_key_callback(ord("F"), toggle_flow_lines)
    vis.create_window()

    # Add point clouds and line sets to the visualization window
    vis.add_geometry(pcd)
    vis.add_geometry(pcd2)
    # vis.add_geometry(pcd_flowed1)
    # vis.add_geometry(pcd_flowed2)
    vis.add_geometry(lineset_flow1)
    vis.add_geometry(lineset_flow2)
    vis.add_geometry(lineset_flow_gt)

    # Run the visualization loop
    vis.run()
    vis.destroy_window()
    return status


def process_files(file1: Path, file2: Path):
    data1 = load_npy(file1, verbose=False).item()
    data2 = load_npy(file2, verbose=False).item()

    assert (data1['pc1'] == data2['pc1']).all(
    ), f"Error: {file1} and {file2} do not have the same point clouds 'pc1'"
    assert (data1['pc2'] == data2['pc2']).all(
    ), f"Error: {file1} and {file2} do not have the same point clouds 'pc2'"
    assert (data1['gt_flow'] == data2['gt_flow']).all(
    ), f"Error: {file1} and {file2} do not have the same ground truth flow"

    gt_flow = data1['gt_flow']
    pc1 = data1['pc1']
    pc2 = data1['pc2']
    est_flow1 = data1['est_flow']
    est_flow2 = data2['est_flow']

    flow1_matches_gt = not flows_disagree(gt_flow, est_flow1)
    flow2_matches_gt = not flows_disagree(gt_flow, est_flow2)
    flow1_matches_flow2 = not flows_disagree(est_flow1, est_flow2)

    dump_file_blacklist = Path() / "dump_file_blacklist.txt"
    dump_file_blacklist.touch(exist_ok=True)
    blacklist = load_txt(dump_file_blacklist, verbose=False).splitlines()
    dump_file_whitelist = Path() / "dump_file_whitelist.txt"
    dump_file_whitelist.touch(exist_ok=True)
    whitelist = load_txt(dump_file_whitelist, verbose=False).splitlines()

    if str(file1) in blacklist:
        return

    if (not flow1_matches_gt) and (flow2_matches_gt) and (
            not flow1_matches_flow2):
        print(f"{file1} does not match ground truth but {file2} does")
        status = visualize_pc_and_flows(pc1, pc2, gt_flow, est_flow1,
                                        est_flow2)
        if status["skip"]:
            blacklist.append(str(file1))
            save_txt(dump_file_blacklist, "\n".join(blacklist))
        elif status["save"]:
            whitelist.append(str(file1))
            save_txt(dump_file_whitelist, "\n".join(whitelist))


def create_files_pairs_to_process():
    folder1_dir_list = sorted(folder1.glob("*/"))
    folder2_dir_list = sorted(folder2.glob("*/"))

    assert len(folder1_dir_list) > 0, f"Error: {folder1} is empty"
    assert len(folder2_dir_list) > 0, f"Error: {folder2} is empty"

    assert [e.name for e in folder1_dir_list] == [
        e.name for e in folder2_dir_list
    ], f"Error: {folder1} and {folder2} do not have the same subfolders"

    files_to_process = []
    for folder1_dir, folder2_dir in zip(folder1_dir_list, folder2_dir_list):
        folder1_file_list = sorted(folder1_dir.glob("*.npy"))
        folder2_file_list = sorted(folder2_dir.glob("*.npy"))

        assert [e.name for e in folder1_file_list] == [
            e.name for e in folder2_file_list
        ], f"Error: {folder1_dir} and {folder2_dir} do not have the same files"

        for file1, file2 in zip(folder1_file_list, folder2_file_list):
            files_to_process.append((file1, file2))
    return files_to_process


files_to_process = create_files_pairs_to_process()

print(f"Found {len(files_to_process)} files to process")

if args.cpus <= 1:
    for file1, file2 in tqdm.tqdm(files_to_process):
        process_files(file1, file2)
else:
    with multiprocessing.Pool(processes=args.cpus) as pool:
        pool.starmap(process_files, files_to_process)
