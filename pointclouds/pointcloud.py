import numpy as np

from .se3 import SE3

class PointCloud():

    def __init__(self, points : np.ndarray) -> None:

        if not isinstance(points, np.ndarray):
            raise TypeError('points must be a numpy.ndarray')
        assert points.ndim == 2, 'points must be a 2D array'
        assert points.shape[1] == 3, 'points must be a Nx3 array'
        self.points = points

    def __len__(self):
        return self.points.shape[0]

    def __repr__(self) -> str:
        return f'PointCloud with {len(self)} points'

    def __getitem__(self, idx):
        return self.points[idx]

    def transform(self, se3 : SE3) -> 'PointCloud':
        assert isinstance(se3, SE3)
        return PointCloud(se3.transform_points(self.points))

