from sklearn.neighbors import NearestNeighbors
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np

from waymo_open_dataset import dataset_pb2
import json

parser = argparse.ArgumentParser()
parser.add_argument("waymo_directory",
                    type=Path,
                    help="Path to Waymo Open directory.")
parser.add_argument("output_directory",
                    type=Path,
                    help="Path to output directory.")
parser.add_argument(
    '--cells_per_meter',
    type=int,
    default=10.0 / 3.0,
    help=
    "Cells per meter for discritization. Default is Argoverse 2 Sensor dataset default is 30cm (10/3 cells per meter)"
)
parser.add_argument('--num_neighbors',
                    type=int,
                    default=20,
                    help="Number of neighbors to use to compute height")
parser.add_argument('--cpus',
                    type=int,
                    default=multiprocessing.cpu_count(),
                    help="Number of cpus to use for parallel processing")

args = parser.parse_args()

assert args.cells_per_meter > 0, "Cells per meter must be positive"
assert args.num_neighbors > 0, "Number of neighbors must be greater than zero"
assert args.waymo_directory.is_dir(
), f"{args.waymo_directory} is not a directory"

print("Waymo directory:", args.waymo_directory)
print("Output directory:", args.output_directory)


def build_knn(points, num_neighbors):
    return NearestNeighbors(n_neighbors=num_neighbors,
                            radius=20,
                            leaf_size=num_neighbors).fit(points)


def build_global_grid(points, cells_per_meter):
    xs, ys, _ = zip(*points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    grid_max_global_frame = np.array([max_x, max_y])
    grid_min_global_frame = np.array([min_x, min_y])

    area_grid_global_frame = grid_max_global_frame - grid_min_global_frame
    grid_shape = np.ceil(
        area_grid_global_frame * cells_per_meter).astype(int) + 1

    def global_to_grid_float(pts):
        assert (pts <= grid_max_global_frame
                ).all(), f"({pts} <= {grid_max_global_frame})"
        assert (pts >= grid_min_global_frame
                ).all(), f"({pts} >= {grid_min_global_frame})"
        relative_to_grid_origin = pts - grid_min_global_frame
        floating_point_grid_coordinate = relative_to_grid_origin * cells_per_meter
        return floating_point_grid_coordinate

    def global_to_grid_index(pts):
        coords = global_to_grid_float(pts)
        return np.round(coords).astype(int)

    return grid_shape, global_to_grid_index, global_to_grid_float, grid_min_global_frame


def render_heightmap(points, cells_per_meter, num_neighbors):

    # We construct this full heightmap with 0, 0 at xy_min_offset in the global coordinate frame.
    grid_shape, global_to_grid_index, global_to_grid_float, grid_min_global_frame = build_global_grid(
        points, cells_per_meter)

    polygon_grid_points = np.array([(*global_to_grid_index(p[:2]), p[2])
                                    for p in points])
    mean_z = np.mean([e[2] for e in polygon_grid_points])
    knn = build_knn(polygon_grid_points, num_neighbors)

    # Construct a grid shaped array whose last axis holds the X, Y index value for that square,
    # with the average Z value for the purposes of querying.
    xs_lin = np.arange(0, grid_shape[0], 1)
    ys_lin = np.arange(0, grid_shape[1], 1)
    xs_square = np.expand_dims(np.tile(xs_lin, (grid_shape[1], 1)), 2)
    ys_square = np.expand_dims(np.tile(ys_lin, (grid_shape[0], 1)).T, 2)
    zs_square = np.ones_like(xs_square) * mean_z
    pts_square = np.concatenate((xs_square, ys_square, zs_square), 2)

    # Flatten the pts square into an N x 3 array for querying KNN
    pts_square_shape = pts_square.shape
    pts_line = pts_square.reshape(pts_square_shape[0] * pts_square_shape[1], 3)

    _, indices = knn.kneighbors(pts_line)
    neighbor_values = polygon_grid_points[indices]
    avg_neighbor_z_values = np.mean(neighbor_values[:, :, 2], axis=1)

    # Reshape flattened average Z values back into grid shape.
    grid = avg_neighbor_z_values.reshape(pts_square_shape[0],
                                         pts_square_shape[1])

    return grid, grid_min_global_frame


def save_grid_global_offset(file_path: Path,
                            grid,
                            grid_min_global_frame,
                            cells_per_meter,
                            verbose=False):
    se2 = {
        "R": [1.0, 0.0, 0.0, 1.0],  # Identity rotation matrix flattened
        "t": [-grid_min_global_frame[0], -grid_min_global_frame[1]],
        "s": cells_per_meter
    }

    save_folder = args.output_directory / file_path.parent.name / (
        file_path.stem + "_map")
    save_folder.mkdir(parents=True, exist_ok=True)

    se2_name = "se2.json"
    height_map_name = "ground_height.npy"

    height_map_file = save_folder / height_map_name
    if height_map_file.exists():
        height_map_file.unlink()

    se2_file = save_folder / se2_name
    if se2_file.exists():
        se2_file.unlink()

    np.save(height_map_file, grid.astype(np.float16))
    if verbose:
        print(f"Saving heightmap to {height_map_file}")
    with open(se2_file, 'w') as fp:
        json.dump(se2, fp)


def polygon_to_points(polygon) -> np.ndarray:
    return [np.array([e.x, e.y, e.z]) for e in polygon]


def collect_points(frame: dataset_pb2.Frame) -> np.ndarray:
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
    print("Processing", file_path)
    dataset = tf.data.TFRecordDataset(file_path, compression_type='')

    # Hack because I can't figure out how to extract the first frame from the dataset.
    for data in dataset:
        frame = dataset_pb2.Frame.FromString(bytearray(data.numpy()))
        break

    points = collect_points(frame)
    grid, grid_min_global_frame = render_heightmap(points,
                                                   args.cells_per_meter,
                                                   args.num_neighbors)
    save_grid_global_offset(file_path,
                            grid,
                            grid_min_global_frame,
                            args.cells_per_meter,
                            verbose=True)


work_queue = build_work_queue(args.waymo_directory)
print("Work queue size:", len(work_queue))

num_processes = min(args.cpus, len(work_queue))
Parallel(num_processes)(delayed(process_record)(record)
                        for record in work_queue)
