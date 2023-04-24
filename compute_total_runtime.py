import argparse
from loader_utils import load_npz
from pathlib import Path
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

parser = argparse.ArgumentParser(description='Process some folders.')
parser.add_argument('--cpus', type=int, default=cpu_count(),
                    help='Number of CPUs to use for multiprocessing')
parser.add_argument('--key', type=str, default="delta_time",
                    help='Key to use for measuring delta time')
args = parser.parse_args()

cpus = args.cpus


argoverse_train_dir = Path('/efs/argoverse2/train_nsfp_flow')
argoverse_val_dir = Path('/efs/argoverse2/val_nsfp_flow')
argoverse_chodosh_train_dir = Path('/efs/argoverse2/train_chodosh_flow/')
waymo_open_train_dir = Path('/efs/waymo_open_processed_flow/train_nsfp_flow')


def process_file(file, key="delta_time"):
    npz_data = dict(load_npz(file, verbose=False))
    return npz_data[key]

def process_folder(path : Path):
    delta_time = 0

    process_function = partial(process_file, key=args.key)
    if cpus <= 1:
        # Use a for loop instead of multiprocessing
        subfolders = path.iterdir()
        file_list = [subfolder.glob("*.npz") for subfolder in subfolders]
        file_list = [file for sublist in file_list for file in sublist]
        for file in tqdm.tqdm(file_list):
            delta_time += process_function(file)
    else:
        with Pool(processes=cpus) as pool:
            subfolders = path.iterdir()
            file_list = [subfolder.glob("*.npz") for subfolder in subfolders]
            delta_time_list = pool.map(process_function, [file for sublist in file_list for file in sublist])
            delta_time = sum(delta_time_list)
    return delta_time


# waymo_train = process_folder(waymo_open_train_dir)
# print("Waymo train: ", waymo_train)
# argo_train = process_folder(argoverse_train_dir)
# print("Argo train: ", argo_train)
# argo_val = process_folder(argoverse_val_dir)
# print("Argo val: ", argo_val)
argo_chodosh_train = process_folder(argoverse_chodosh_train_dir)
print("Argo chodosh train: ", argo_chodosh_train)