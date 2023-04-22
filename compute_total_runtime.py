from loader_utils import load_npz
from pathlib import Path
import tqdm
from multiprocessing import Pool, cpu_count

argoverse_train_dir = Path('/efs/argoverse2/train_nsfp_flow')
argoverse_val_dir = Path('/efs/argoverse2/val_nsfp_flow')
waymo_open_train_dir = Path('/efs/waymo_open_processed_flow/train_nsfp_flow')


def process_file(file):
    npz_data = dict(load_npz(file, verbose=False))
    return npz_data["delta_time"]

def process_folder(path : Path):
    delta_time = 0
    with Pool(processes=cpu_count()) as pool:
        subfolders = list(path.iterdir())
        file_list = [list(subfolder.glob("*.npz")) for subfolder in subfolders]
        delta_time_list = pool.map(process_file, [file for sublist in file_list for file in sublist])
        delta_time = sum(delta_time_list)
    return delta_time


waymo_train = process_folder(waymo_open_train_dir)
print("Waymo train: ", waymo_train)
argo_train = process_folder(argoverse_train_dir)
print("Argo train: ", argo_train)
argo_val = process_folder(argoverse_val_dir)
print("Argo val: ", argo_val)