import argparse
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Optional

# Get path to the eval frames file

parser = argparse.ArgumentParser()
parser.add_argument('argoverse_dir',
                    type=Path,
                    help='path to the Argoverse dataset')
parser.add_argument('eval_frames_file',
                    type=Path,
                    help='path to the eval frames file')
parser.add_argument('output_file', type=Path, help='path to the out file')
args = parser.parse_args()

assert args.argoverse_dir.exists(), 'Argoverse directory does not exist'
assert args.eval_frames_file.exists(), 'Eval frames file does not exist'


def process_lines(lines: List[str]) -> List[Tuple[str, int, int]]:

    def process_line(line: str) -> Tuple[str, int, int]:
        line = line.strip()
        line = line.split(',')
        return line[0], int(line[1]), int(line[2])

    return [process_line(line) for line in lines]


# Load txt file
with open(args.eval_frames_file, 'r') as f:
    lines = f.readlines()
configs = process_lines(lines)


def config_to_id_index(config: Tuple[str, int, int]) -> Tuple[str, int]:
    id, start, end = config
    feather_folder = args.argoverse_dir / id / 'sensors' / 'lidar'
    feather_files = sorted(feather_folder.glob('*.feather'))
    feather_names = [int(f.stem) for f in feather_files]
    index = feather_names.index(start)
    assert feather_names[index] == start, 'Start frame not found'
    assert feather_names[index + 1] == end, 'End frame not found'
    return id, index


# Convert configs to id, index
configs = [config_to_id_index(config) for config in configs]

# Write to file
with open(args.output_file, 'w') as f:
    for config in configs:
        f.write(f'{config[0]},{config[1]}\n')
