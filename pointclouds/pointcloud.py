import numpy as np

from .se3 import SE3


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

    def to_fixed_array(self, max_points: int) -> np.ndarray:
        if len(self) > max_points:
            np.random.RandomState(len(self.points)).shuffle(self.points)
            sliced_pts = self.points[:max_points]
            # add a 4th column of 1s to indicate that these points are valid
            return np.pad(sliced_pts, ((0, 0), (0, 1)), constant_values=1)
        else:
            # pad existing points with 0s to indicate that these points are valid
            padded_pts = np.pad(self.points, ((0, 0), (0, 1)),
                                constant_values=1)
            return np.pad(padded_pts, ((0, max_points - len(self)), (0, 0)),
                          constant_values=0)

    @staticmethod
    def from_fixed_array(points) -> 'PointCloud':
        are_valid_points = (points[:, 3] == 1)
        if isinstance(are_valid_points, np.ndarray):
            are_valid_points = are_valid_points.astype(bool)
        else:
            are_valid_points = are_valid_points.bool()
        return PointCloud(points[are_valid_points, :3])

    def to_array(self) -> np.ndarray:
        return self.points

    @staticmethod
    def from_array(points: np.ndarray) -> 'PointCloud':
        return PointCloud(points)

    def mask_points(self, mask: np.ndarray) -> 'PointCloud':
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 1
        assert mask.shape[0] == len(self)
        assert mask.dtype == bool
        return PointCloud(self.points[mask])
