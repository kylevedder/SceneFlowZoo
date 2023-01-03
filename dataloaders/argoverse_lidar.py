import numpy as np
import pandas as pd
from pathlib import Path
from pointclouds import PointCloud, SE3

class ArgoverseSequence():

    def __init__(self, dataset_dir : Path):
        self.dataset_dir = Path(dataset_dir)
        assert self.dataset_dir.is_dir(), f'dataset_dir {dataset_dir} does not exist'
        self.frame_paths = sorted((self.dataset_dir / 'sensors'/ 'lidar').glob('*.feather'))
        assert len(self.frame_paths) > 0, f'no frames found in {self.dataset_dir}'
        self.frame_infos = pd.read_feather(self.dataset_dir / 'city_SE3_egovehicle.feather')
        assert len(self.frame_paths) == len(self.frame_infos), f'frame_paths {len(self.frame_paths)} and frame_infos {len(self.frame_infos)} do not match'

        # Verify that the timestamps match for infos and frames
        infos_timestamps = self.frame_infos['timestamp_ns'].values
        frame_timestamps = np.array([int(e.name.split('.')[0]) for e in self.frame_paths])
        assert np.all(infos_timestamps == frame_timestamps), f'frame_paths and frame_infos do not match'

    def __repr__(self) -> str:
        return f'ArgoverseSequence with {len(self)} frames'

    def __len__(self):
        return len(self.frame_paths)


    def _load_pc(self, idx) -> PointCloud:
        frame_path = self.frame_paths[idx]
        frame_content = pd.read_feather(frame_path)
        xs = frame_content['x'].values
        ys = frame_content['y'].values
        zs = frame_content['z'].values
        points = np.stack([xs, ys, zs], axis=1)
        return PointCloud(points)

    def _load_pose(self, idx) -> SE3:
        frame_info = self.frame_infos.iloc[idx]
        se3 = SE3.from_rot_w_x_y_z_translation_x_y_z(
            frame_info['qw'],
            frame_info['qx'],
            frame_info['qy'],
            frame_info['qz'],
            frame_info['tx_m'],
            frame_info['ty_m'],
            frame_info['tz_m'])
        return se3


    def __getitem__(self, idx) -> (PointCloud, SE3):
        pc = self._load_pc(idx)
        pose = self._load_pose(idx)
        return pc, pose

    def __iter__(self):
        _, init_pose = self[0]
        for idx in range(len(self)):
            pc, pose = self[idx]
            yield pc, (init_pose.inverse().compose(pose))


class ArgoverseSequenceLoader():

    def __init__(self, sequence_dir : Path):
        self.dataset_dir = Path(sequence_dir)
        assert self.dataset_dir.is_dir(), f'dataset_dir {sequence_dir} does not exist'
        self.log_lookup = {e.name : e for e in self.dataset_dir.glob('*/')}

    def get_sequence_ids(self):
        return sorted(self.log_lookup.keys())

    def load_sequence(self, log_id : str) -> ArgoverseSequence:
        assert log_id in self.log_lookup, f'log_id {log_id} does not exist'
        log_dir = self.log_lookup[log_id]
        assert log_dir.is_dir(), f'log_id {log_id} does not exist'
        return ArgoverseSequence(log_dir)


# def load_argoverse(dataset_dir, log_id):
    # sdb = SynchronizationDB(dataset_dir, collect_single_log_id=log_id)
