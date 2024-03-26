import argparse
from pathlib import Path
import numpy as np
from core_utils import load_npz, save_npz
import joblib
import multiprocessing

# Get path to existing nsfp flow folder
# Get path to new odom flow folder

parser = argparse.ArgumentParser()
parser.add_argument('nsfp_dir', type=Path, help='path to the nsfp flow folder')
parser.add_argument('odom_dir', type=Path, help='path to the odom flow folder')
args = parser.parse_args()

assert args.nsfp_dir.exists(), 'nsfp directory does not exist'
args.odom_dir.mkdir(exist_ok=True)

sequence_dirs = sorted(args.nsfp_dir.glob('*/'))


def process_sequence(sequence_dir: Path):
    # Collect the npz files
    npz_files = sorted(sequence_dir.glob('*.npz'))
    # Create a new folder for the odom flows
    odom_dir = args.odom_dir / sequence_dir.name
    odom_dir.mkdir(exist_ok=True)
    # For each npz file, load the data and save the odom flow
    for npz_file in npz_files:
        data = dict(load_npz(npz_file, verbose=False))
        data['flow'] = np.zeros_like(data['flow'])
        save_npz(odom_dir / npz_file.name, data, verbose=False)


# Parallel process the sequences
joblib.Parallel(n_jobs=multiprocessing.cpu_count() - 1)(joblib.delayed(process_sequence)(sequence_dir)
                          for sequence_dir in sequence_dirs)
