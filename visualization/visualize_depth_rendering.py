import argparse
import torch
from pathlib import Path


from models.whole_batch_optimization.checkpointing.model_loader import OptimCheckpointModelLoader
from models.mini_batch_optimization import EulerFlowOccFlowModel
from models import ForwardMode
from bucketed_scene_flow_eval.datastructures import EgoLidarDistance
from bucketed_scene_flow_eval.utils import save_pickle, load_pickle
from bucketed_scene_flow_eval.datastructures import O3DVisualizer, PointCloud, SE3
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from dataclasses import dataclass, field


@dataclass
class DistanceBucket:
    gt_distances: list[float] = field(default_factory=list)
    estimated_distances: list[float] = field(default_factory=list)

    def add_gt_distances(self, gt_distances: np.ndarray):
        self.gt_distances.extend(gt_distances)

    def add_estimated_distances(self, estimated_distances: np.ndarray):
        self.estimated_distances.extend(estimated_distances)

    def __len__(self):
        return len(self.gt_distances)

    def get_errors(self):
        return np.array(self.estimated_distances) - np.array(self.gt_distances)

    def get_l1_errors(self):
        return np.abs(self.get_errors())

    def get_mean_l1_error(self):
        return np.mean(self.get_l1_errors())

    def get_average_gt_distance(self):
        return np.mean(self.gt_distances)

    def get_average_estimated_distance(self):
        return np.mean(self.estimated_distances)


class DistanceBuckets:

    def __init__(self, bucket_edges: list[float]):
        self.bucket_edges = bucket_edges
        self.buckets: list[DistanceBucket] = [
            DistanceBucket() for _ in range(len(bucket_edges) - 1)
        ]

    def add_distances(self, gt_distances: np.ndarray, estimated_distances: np.ndarray):
        for idx, edge in enumerate(self.bucket_edges[:-1]):
            mask = (gt_distances >= edge) & (gt_distances < self.bucket_edges[idx + 1])
            distance_bucket = self.buckets[idx]
            distance_bucket.add_gt_distances(gt_distances[mask])
            distance_bucket.add_estimated_distances(estimated_distances[mask])

    def get_bucket_data(self, bucket_idx: int) -> tuple[float, float, DistanceBucket]:
        bucket = self.buckets[bucket_idx]
        lower_edge = self.bucket_edges[bucket_idx]
        upper_edge = self.bucket_edges[bucket_idx + 1]
        return lower_edge, upper_edge, bucket

    def get_bucket_datas(self) -> list[tuple[float, float, DistanceBucket]]:
        return [self.get_bucket_data(idx) for idx in range(len(self.buckets))]


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
    model: EulerFlowOccFlowModel
    assert isinstance(
        model, EulerFlowOccFlowModel
    ), f"Expected EulerFlowOccFlowModel, got {type(model)}"

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

    distance_buckets = DistanceBuckets([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 100])
    for idx, ego_lidar_distance in enumerate(tqdm.tqdm(output_occ_list)):
        source_pc = PointCloud(full_sequence.get_full_ego_pc(idx).cpu().numpy())
        is_valid_mask = full_sequence.get_full_pc_mask(idx).cpu().numpy()
        gt_ray_lengths = np.linalg.norm(source_pc.points, axis=1)
        estimated_distances = ego_lidar_distance.distances

        valid_gt_ray_lengths = gt_ray_lengths[is_valid_mask]
        valid_estimated_distances = estimated_distances[is_valid_mask]

        distance_buckets.add_distances(valid_gt_ray_lengths, valid_estimated_distances)

    bucket_datas = distance_buckets.get_bucket_datas()

    # Make subplot for each bucket
    fig, axs = plt.subplots(len(bucket_datas), 1)

    for idx, (lower_edge, upper_edge, bucket) in enumerate(bucket_datas):
        l1_error_mean = bucket.get_mean_l1_error()
        bucket_errors = bucket.get_errors()

        gt_average_distance = bucket.get_average_gt_distance()

        axs[idx].hist(bucket_errors, bins=100)
        axs[idx].set_title(
            f"Bucket: {lower_edge} - {upper_edge}: L1 Error Mean: {l1_error_mean:.2f}; N = {len(bucket)}; Percentage of GT Dist: {l1_error_mean / gt_average_distance * 100:.2f}%"
        )

    # make tight
    plt.tight_layout()
    plt.show()

    exit(0)

    l1_errors_array = np.concatenate(l1_errors, axis=0)
    abs_l1_errors_array = np.abs(l1_errors_array)
    percentile_90 = np.percentile(abs_l1_errors_array, 90)
    percentile_75 = np.percentile(abs_l1_errors_array, 75)
    tolerance_threshold = 1

    for idx, ego_lidar_distance in enumerate(output_occ_list):

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

    plt.hist(abs_l1_errors_array, bins=100)
    # Add vertical bar at 75th and 90th percentile
    plt.axvline(np.percentile(abs_l1_errors_array, 75), color="r", linestyle="--")
    plt.axvline(np.percentile(abs_l1_errors_array, 90), color="r", linestyle="--")
    plt.title(f"L1 Ray Error Histogram. Avg: {np.mean(abs_l1_errors_array):.2f}")
    plt.xlabel("L1 Ray Error (meters)")
    plt.ylabel("Count")
    plt.show()

    plt.hist(l1_errors_array, bins=100)
    # Add vertical bar at 75th and 90th percentile
    plt.title(f"Est - GT Histogram. Avg: {np.mean(l1_errors_array):.2f}")
    plt.xlabel("Error (meters)")
    plt.ylabel("Count")
    plt.show()

    vis.run()

    breakpoint()


if __name__ == "__main__":
    main()
