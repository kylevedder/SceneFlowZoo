from pathlib import Path
import numpy as np
import tqdm
import joblib
import multiprocessing
from loader_utils import save_pickle
import open3d as o3d

from dataloaders import ArgoverseRawSequenceLoader

argoverse_data = Path("/efs/argoverse2")

PARALLELIZE = True


def get_pc_sizes(seq):
    pc_sizes = []
    for idx in range(len(seq)):
        frame = seq.load(idx, idx)
        pc = frame['relative_pc']
        pc = pc.within_region(-51.2, 51.2, -51.2, 51.2, -3, 1)
        num_points = len(pc)
        if num_points < 100:
            print(seq.log_id, frame['log_idx'], num_points)
        pc_sizes.append(num_points)
    return pc_sizes


seq_loader = ArgoverseRawSequenceLoader(argoverse_data / "val")
seq_ids = seq_loader.get_sequence_ids()

if PARALLELIZE:
    # Use joblib to parallelize the loading of the sequences
    pc_size_lst = joblib.Parallel(n_jobs=multiprocessing.cpu_count() - 1)(
        joblib.delayed(get_pc_sizes)(seq_loader.load_sequence(seq_id))
        for seq_id in tqdm.tqdm(seq_ids))
    pc_size_lst = [item for sublist in pc_size_lst for item in sublist]
else:
    pc_size_lst = []
    for seq_id in tqdm.tqdm(seq_ids):
        seq = seq_loader.load_sequence(seq_id)
        pc_sizes = get_pc_sizes(seq)
        pc_size_lst.extend(pc_sizes)

save_pickle("validation_results/validation_pointcloud_point_count.pkl",
            pc_size_lst)

print(np.mean(pc_size_lst), np.std(pc_size_lst), np.median(pc_size_lst))
