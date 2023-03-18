from pathlib import Path
import argparse
import numpy as np

# Path to the NSFP flows
parser = argparse.ArgumentParser()
parser.add_argument('nsfp_flows_dir', type=Path, help='path to the NSFP flows')
parser.add_argument('expected_num_flows',
                    type=int,
                    help='expected number of flows')
args = parser.parse_args()

assert args.nsfp_flows_dir.exists(), 'NSFP flows dir does not exist'
assert args.expected_num_flows > 0, 'Expected number of flows must be positive'

subfolders = sorted([f for f in args.nsfp_flows_dir.glob('*') if f.is_dir()])

# Check that all subfolders have the expected number of flows
for subfolder in subfolders:
    num_flows = len([f for f in subfolder.glob('*.npz') if f.is_file()])
    if num_flows != args.expected_num_flows:
        print(subfolder.name, num_flows)
