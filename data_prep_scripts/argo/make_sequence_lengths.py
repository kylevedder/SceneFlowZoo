from pathlib import Path
import argparse
import json

# Take argument for input sequence_folder

parser = argparse.ArgumentParser()
parser.add_argument("sequence_folder", type=Path)
args = parser.parse_args()

sequence_folder: Path = args.sequence_folder
assert sequence_folder.is_dir(), f"Sequence folder {sequence_folder} does not exist."

# Count the number of files under sequence_folder/*/sensors/lidar/ and save to sequence_folder_sequence_lengths.txt

input_folders = sorted(sequence_folder.glob(f"*/sensors/lidar/"))
output_file = sequence_folder.parent / f"{sequence_folder.stem}_sequence_lengths.json"

print(
    f"Counting the number of files under {sequence_folder}/*/sensors/lidar/ and saving to {output_file}"
)


def process_folder(folder: Path):
    files = list(folder.glob("*.feather"))
    return len(files)


name_lookup = {folder.parent.parent.stem: process_folder(folder) for folder in input_folders}

print(f"Writing sequence lengths to {output_file}")

with open(output_file, "w") as f:
    json.dump(name_lookup, f, indent=4)
    # Write newline at end of file
    f.write("\n")
