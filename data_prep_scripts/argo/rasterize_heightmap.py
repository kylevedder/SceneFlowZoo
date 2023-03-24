import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from sklearn.neighbors import NearestNeighbors
import numpy as np
import json
from pathlib import Path
import multiprocessing
import matplotlib.pyplot as plt
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm

import argparse


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


def collect_points(drivable_areas):
    all_points = []

    for key in drivable_areas.keys():
        points = [(e['x'], e['y'], e['z'])
                  for e in drivable_areas[key]['area_boundary']]
        all_points.append(points)

    return all_points


def get_road_polygon_mask_arr(polygon_list, mask_shape, global_to_grid_float,
                              cells_per_meter, meters_beyond_poly_edge):
    mask_img = Image.new('1', (mask_shape[1], mask_shape[0]), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    for polygon in polygon_list:
        grid_pts = [tuple(global_to_grid_float(p[:2])) for p in polygon]
        mask_draw.polygon(grid_pts, fill=1, outline=1)

    blur_radius = int(np.ceil(meters_beyond_poly_edge * cells_per_meter))
    mask_img = mask_img.convert('L')
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return (np.array(mask_img) > 0).astype(int)


def render_heightmap(drivable_areas, cells_per_meter, meters_beyond_poly_edge,
                     num_neighbors):

    polygon_list = collect_points(drivable_areas)

    flattened_poly_points = [e for e_list in polygon_list for e in e_list]

    # We construct this full heightmap with 0, 0 at xy_min_offset in the global coordinate frame.
    grid_shape, global_to_grid_index, global_to_grid_float, grid_min_global_frame = build_global_grid(
        flattened_poly_points, cells_per_meter)

    polygon_grid_points = np.array([(*global_to_grid_index(p[:2]), p[2])
                                    for p in flattened_poly_points])
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

    if meters_beyond_poly_edge is not None:
        road_mask_array = get_road_polygon_mask_arr(polygon_list, grid.shape,
                                                    global_to_grid_float,
                                                    cells_per_meter,
                                                    meters_beyond_poly_edge)
        grid[~road_mask_array] = np.NaN
    return grid, grid_min_global_frame


def save_grid_global_offset(sequence_folder: Path, grid, grid_min_global_frame,
                            cells_per_meter):
    se2 = {
        "R": [1.0, 0.0, 0.0, 1.0],  # Identity rotation matrix flattened
        "t": [-grid_min_global_frame[0], -grid_min_global_frame[1]],
        "s": cells_per_meter
    }

    se2_name = f"{sequence_folder.name}___img_Sim2_city.json"
    height_map_name = f"{sequence_folder.name}_ground_height_surface____.npy"

    height_map_file = sequence_folder / "map" / height_map_name
    if height_map_file.exists():
        height_map_file.unlink()

    se2_file = sequence_folder / "map" / se2_name
    if se2_file.exists():
        se2_file.unlink()

    np.save(height_map_file, grid.astype(np.float16))
    with open(se2_file, 'w') as fp:
        json.dump(se2, fp)


def process_sequence_folder(sequence_folder: Path,
                            cells_per_meter=10.0 / 3.0,
                            meters_beyond_poly_edge=None,
                            num_neighbors=20):
    sequence_folder = Path(sequence_folder)
    map_files = list((sequence_folder / "map").glob("log_map_archive_*.json"))
    assert len(
        map_files
    ) == 1, f"Expected 1 map file, got {len(map_files)} in {sequence_folder / 'map'}"
    map_file = map_files[0]
    with open(map_file) as f:
        map_json = json.load(f)

    grid, grid_min_global_frame = render_heightmap(map_json['drivable_areas'],
                                                   cells_per_meter,
                                                   meters_beyond_poly_edge,
                                                   num_neighbors)
    save_grid_global_offset(sequence_folder, grid, grid_min_global_frame,
                            cells_per_meter)


def build_work_queue(root_path: Path):
    root_path = Path(root_path)
    root_folders = [
        e for e in root_path.glob("*/")
        if e.is_dir() and e.name in ["train", "val"]
    ]
    work_queue = []
    for rf in root_folders:
        work_queue.extend([e for e in rf.glob("*/") if e.is_dir()])
    return work_queue


parser = argparse.ArgumentParser()
parser.add_argument("argoverse_directory",
                    help="Path to Argoverse 2 directory.")
parser.add_argument(
    '--meters_beyond_poly_edge',
    type=float,
    default=5.0,
    help="Meters beyond the poly road edge to project out the height map."
    "For the Argoverse 2 sensor dataset, the heightmap extends 5 meters outside the road geometry polygon. "
    "For values less than 0, this will generate the height map for the entire scene."
)
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

argoverse_directory = Path(args.argoverse_directory)
assert argoverse_directory.is_dir(
), f"{argoverse_directory} is not a directory"

cells_per_meter = args.cells_per_meter
assert cells_per_meter > 0, f"Cells per meter must be positive (got {cells_per_meter})"

meters_beyond_poly_edge = args.meters_beyond_poly_edge if args.meters_beyond_poly_edge >= 0 else None

num_neighbors = args.num_neighbors
assert num_neighbors > 0, f"Number of neighbors must be greater than zero (got {num_neighbors})"

work_queue = build_work_queue(argoverse_directory)

print("Work queue size:", len(work_queue))

Parallel(args.cpus)(delayed(process_sequence_folder)(
    sequence_folder=e,
    cells_per_meter=cells_per_meter,
    meters_beyond_poly_edge=meters_beyond_poly_edge,
    num_neighbors=num_neighbors) for e in tqdm(work_queue))

# with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#     pool.map(
#         partial(process_sequence_folder,
#                 cells_per_meter=cells_per_meter,
#                 meters_beyond_poly_edge=meters_beyond_poly_edge,
#                 num_neighbors=num_neighbors), work_queue)
