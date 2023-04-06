import numpy as np

from .se3 import SE3


def to_fixed_array(array: np.ndarray,
                   max_len: int,
                   pad_val=np.nan) -> np.ndarray:
    if len(array) > max_len:
        np.random.RandomState(len(array)).shuffle(array)
        sliced_pts = array[:max_len]
        return sliced_pts
    else:
        pad_tuples = [(0, max_len - len(array))]
        for _ in range(array.ndim - 1):
            pad_tuples.append((0, 0))
        return np.pad(array, pad_tuples, constant_values=pad_val)


def from_fixed_array(array: np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray):
        are_valid_points = np.logical_not(np.isnan(array[:, 0]))
        are_valid_points = are_valid_points.astype(bool)
    else:
        import torch
        are_valid_points = torch.logical_not(torch.isnan(array[:, 0]))
        are_valid_points = are_valid_points.bool()
    return array[are_valid_points]


class PointCloud():

    def __init__(self, points: np.ndarray) -> None:
        assert points.ndim == 2, 'points must be a 2D array'
        assert points.shape[1] == 3, 'points must be a Nx3 array'
        self.points = points

    def __len__(self):
        return self.points.shape[0]

    def __repr__(self) -> str:
        return f'PointCloud with {len(self)} points'

    def __getitem__(self, idx):
        return self.points[idx]

    def transform(self, se3: SE3) -> 'PointCloud':
        assert isinstance(se3, SE3)
        return PointCloud(se3.transform_points(self.points))

    def translate(self, translation: np.ndarray) -> 'PointCloud':
        assert translation.shape == (3, )
        return PointCloud(self.points + translation)

    def flow(self, flow: np.ndarray) -> 'PointCloud':
        assert flow.shape == self.points.shape, f"flow shape {flow.shape} must match point cloud shape {self.points.shape}"
        return PointCloud(self.points + flow)

    def to_fixed_array(self, max_points: int) -> np.ndarray:
        return to_fixed_array(self.points, max_points)

    def matched_point_ditsance(self, other: 'PointCloud') -> np.ndarray:
        assert len(self) == len(other)
        return np.linalg.norm(self.points - other.points, axis=1)

    @staticmethod
    def from_fixed_array(points) -> 'PointCloud':
        return PointCloud(from_fixed_array(points))

    def to_array(self) -> np.ndarray:
        return self.points

    @staticmethod
    def from_array(points: np.ndarray) -> 'PointCloud':
        return PointCloud(points)

    def mask_points(self, mask: np.ndarray) -> 'PointCloud':
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 1
        if mask.dtype == np.bool:
            assert mask.shape[0] == len(self)
        else:
            in_bounds = mask < len(self)
            assert np.all(
                in_bounds
            ), f"mask values must be in bounds, got {(~in_bounds).sum()} indices not in bounds out of {len(self)} points"

        return PointCloud(self.points[mask])

    def within_region(self, x_min, x_max, y_min, y_max, z_min,
                      z_max) -> 'PointCloud':
        mask = np.logical_and(self.points[:, 0] < x_max,
                              self.points[:, 0] > x_min)
        mask = np.logical_and(mask, self.points[:, 1] < y_max)
        mask = np.logical_and(mask, self.points[:, 1] > y_min)
        mask = np.logical_and(mask, self.points[:, 2] < z_max)
        mask = np.logical_and(mask, self.points[:, 2] > z_min)
        return self.mask_points(mask)

    @property
    def shape(self) -> tuple:
        return self.points.shape
