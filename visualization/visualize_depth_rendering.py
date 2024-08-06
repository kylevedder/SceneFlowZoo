import argparse
import torch
from pathlib import Path


from models.whole_batch_optimization.checkpointing.model_loader import OptimCheckpointModelLoader
from models.mini_batch_optimization import GigachadOccFlowModel
from models import ForwardMode
from bucketed_scene_flow_eval.datastructures import EgoLidarDistance
from bucketed_scene_flow_eval.utils import save_pickle, load_pickle
from bucketed_scene_flow_eval.datastructures import O3DVisualizer, PointCloud, SE3
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("sequence_id", type=str)
    parser.add_argument(
        "--sequence_id_to_length",
        type=Path,
        default=Path("data_prep_scripts/argo/av2_test_sizes.json"),
    )
    args = parser.parse_args()

    model_loader = OptimCheckpointModelLoader.from_checkpoint_dirs(
        args.config, args.checkpoint_root, args.sequence_id, args.sequence_id_to_length
    )
    model, full_sequence = model_loader.load_model()
    model: GigachadOccFlowModel
    assert isinstance(
        model, GigachadOccFlowModel
    ), f"Expected GigachadOccFlowModel, got {type(model)}"

    output_cache_path = (
        Path("/tmp/occ_flow_cache/")
        / args.config.stem
        / args.sequence_id
        / args.checkpoint_root.stem
        / "output_cache.pkl"
    )

    if not output_cache_path.exists():
        with torch.inference_mode():
            with torch.no_grad():
                (output,) = model(
                    ForwardMode.VAL, [full_sequence.detach().requires_grad_(False)], None
                )

        output_occ_list: list[EgoLidarDistance] = output.to_ego_lidar_distance_list()

        output_cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(output_cache_path, output_occ_list)
    else:
        output_occ_list = load_pickle(output_cache_path)

    assert len(output_occ_list) + 1 == len(
        full_sequence
    ), f"Expected {len(full_sequence) - 1} outputs, got {len(output_occ_list)}"

    # Ensure that all distances are positive. Negative distances are clearly wrong.
    for ego_lidar_distance in output_occ_list:
        assert np.all(ego_lidar_distance.distances >= 0), "Expected all distances to be positive"

    vis = O3DVisualizer()

    l1_errors: list[float] = []
    for idx, ego_lidar_distance in enumerate(output_occ_list):
        source_pc = PointCloud(full_sequence.get_full_ego_pc(idx).cpu().numpy())
        is_valid_mask = full_sequence.get_full_pc_mask(idx).cpu().numpy()
        gt_ray_lengths = np.linalg.norm(source_pc.points, axis=1)
        estimated_distances = ego_lidar_distance.distances
        current_frame_l1_ray_error = (
            estimated_distances[is_valid_mask] - gt_ray_lengths[is_valid_mask]
        )
        l1_errors.append(current_frame_l1_ray_error)

    l1_errors_array = np.concatenate(l1_errors, axis=0)
    abs_l1_errors_array = np.abs(l1_errors_array)
    percentile_90 = np.percentile(abs_l1_errors_array, 90)
    percentile_75 = np.percentile(abs_l1_errors_array, 75)
    tolerance_threshold = 1

    for idx, ego_lidar_distance in enumerate(output_occ_list):

        if idx > 75 or idx < 75:
            continue

        source_pc = PointCloud(full_sequence.get_full_ego_pc(idx).cpu().numpy())
        is_valid_mask = full_sequence.get_full_pc_mask(idx).cpu().numpy()

        gt_ray_lengths = np.linalg.norm(source_pc.points, axis=1)

        normalized_gt_rays = source_pc.points / gt_ray_lengths[:, None]
        estimated_distances = ego_lidar_distance.distances

        current_frame_l1_ray_error = np.abs(
            estimated_distances[is_valid_mask] - gt_ray_lengths[is_valid_mask]
        )

        estimated_rays = normalized_gt_rays * estimated_distances[:, None]

        estimated_pc = PointCloud(estimated_rays)

        sensor_to_ego, ego_to_global = full_sequence.get_pc_transform_matrices(idx)
        sensor_to_ego = SE3.from_array(sensor_to_ego.cpu().numpy())
        ego_to_global = SE3.from_array(ego_to_global.cpu().numpy())
        sensor_to_global = ego_to_global @ sensor_to_ego

        masked_source_pc = source_pc.mask_points(is_valid_mask)
        masked_estimated_pc = estimated_pc.mask_points(is_valid_mask)

        masked_source_pc_in_tolerance = masked_source_pc.mask_points(
            current_frame_l1_ray_error < tolerance_threshold
        )
        masked_source_pc_out_of_tolerance = masked_source_pc.mask_points(
            current_frame_l1_ray_error >= tolerance_threshold
        )

        vis.add_pointcloud(masked_source_pc_in_tolerance, pose=sensor_to_global, color=[0, 0, 1])

        vis.add_pointcloud(
            masked_source_pc_out_of_tolerance, pose=sensor_to_global, color=[0, 1, 1]
        )

        masked_estimated_pc_in_tolerance = masked_estimated_pc.mask_points(
            current_frame_l1_ray_error < tolerance_threshold
        )
        masked_estimated_pc_out_of_tolerance = masked_estimated_pc.mask_points(
            current_frame_l1_ray_error >= tolerance_threshold
        )

        vis.add_pointcloud(masked_estimated_pc_in_tolerance, pose=sensor_to_global, color=[0, 1, 0])
        vis.add_pointcloud(
            masked_estimated_pc_out_of_tolerance, pose=sensor_to_global, color=[1, 0, 0]
        )

        # Add lineset to ray trace out of tolerance points
        randomly_sampled_out_of_tolerance_mask = np.zeros(
            len(masked_source_pc_out_of_tolerance), dtype=bool
        )
        randomly_subsampled_indices = np.random.choice(
            np.arange(len(masked_source_pc_out_of_tolerance)), size=20, replace=False
        )
        randomly_sampled_out_of_tolerance_mask[randomly_subsampled_indices] = True

        masked_source_pc_out_of_tolerance_sampled = masked_source_pc_out_of_tolerance.mask_points(
            randomly_sampled_out_of_tolerance_mask
        )
        masked_source_pc_out_of_tolerance_sampled_origin = PointCloud(
            np.zeros_like(masked_source_pc_out_of_tolerance_sampled.points)
        )

        masked_source_pc_out_of_tolerance_sampled = (
            masked_source_pc_out_of_tolerance_sampled.transform(sensor_to_global)
        )
        masked_source_pc_out_of_tolerance_sampled_origin = (
            masked_source_pc_out_of_tolerance_sampled_origin.transform(sensor_to_global)
        )

        vis.add_lineset(
            masked_source_pc_out_of_tolerance_sampled_origin,
            masked_source_pc_out_of_tolerance_sampled,
            color=[0, 0, 0],
        )

    # plt.hist(abs_l1_errors_array, bins=100)
    # # Add vertical bar at 75th and 90th percentile
    # plt.axvline(np.percentile(abs_l1_errors_array, 75), color="r", linestyle="--")
    # plt.axvline(np.percentile(abs_l1_errors_array, 90), color="r", linestyle="--")
    # plt.title(f"L1 Ray Error Histogram. Avg: {np.mean(abs_l1_errors_array):.2f}")
    # plt.xlabel("L1 Ray Error (meters)")
    # plt.ylabel("Count")
    # plt.show()

    # plt.hist(l1_errors_array, bins=100)
    # # Add vertical bar at 75th and 90th percentile
    # plt.title(f"Est - GT Histogram. Avg: {np.mean(l1_errors_array):.2f}")
    # plt.xlabel("Error (meters)")
    # plt.ylabel("Count")
    # plt.show()

    vis.run()

    breakpoint()


if __name__ == "__main__":
    main()
