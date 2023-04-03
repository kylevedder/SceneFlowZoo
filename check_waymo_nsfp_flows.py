import argparse
from pathlib import Path

# Get path to the Waymo dataset

parser = argparse.ArgumentParser()
parser.add_argument('waymo_dir', type=Path, help='path to the Waymo dataset')
args = parser.parse_args()

training_dir = args.waymo_dir / 'training'

flow_label_dir = args.waymo_dir / 'train_nsfp_flow'

assert training_dir.exists(), 'Training directory does not exist'

training_subfolders = sorted(training_dir.glob('segment*'))

flow_subfolders = sorted(flow_label_dir.glob('segment*'))

assert len(
    training_subfolders
) == 798, f'Expected 798 training subfolders, got {len(training_subfolders)}'

train_subfolder_names = [e.name for e in training_subfolders]

valid_flow_folders = []
for subfolder in flow_subfolders:
    num_flows = sorted(subfolder.glob('*.npz'))
    idx = train_subfolder_names.index(subfolder.name)
    if len(num_flows) >= 196:
        valid_flow_folders.append(subfolder)

valid_flow_folder_names = sorted([e.name for e in valid_flow_folders])

# Get all the folders in the training subset that aren't in flow subfolders
missing_folder_names = sorted(
    set(train_subfolder_names) - set(valid_flow_folder_names))

print("Missing folder names", missing_folder_names)