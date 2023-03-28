import argparse
from pathlib import Path

# Get path to the Waymo dataset

parser = argparse.ArgumentParser()
parser.add_argument('waymo_dir', type=Path, help='path to the Waymo dataset')
args = parser.parse_args()

training_dir = args.waymo_dir / 'training'
validation_dir = args.waymo_dir / 'validation'

assert training_dir.exists(), 'Training directory does not exist'
assert validation_dir.exists(), 'Validation directory does not exist'

training_subfolders = sorted(training_dir.glob('segment*'))
validation_subfolders = sorted(validation_dir.glob('segment*'))

assert len(training_subfolders) == 798, f'Expected 798 training subfolders, got {len(training_subfolders)}'
assert len(validation_subfolders) == 202, f'Expected 202 validation subfolders, got {len(validation_subfolders)}'

# Check that all subfolders have the expected number of flows
for subfolder in training_subfolders:
    num_flows = subfolder.glob('*.pkl')
    print(subfolder.name, num_flows)

    