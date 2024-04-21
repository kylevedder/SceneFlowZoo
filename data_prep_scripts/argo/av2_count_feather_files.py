import argparse
import json
from pathlib import Path
from bucketed_scene_flow_eval.utils import load_feather, save_json


def feather_file_lengths_in_subfolders(root_folder_path):
    # Convert the input path to a Path object for easy handling
    root_folder = Path(root_folder_path)
    
    # Check if the provided root folder path exists and is a directory
    assert root_folder.is_dir(), f"The path {root_folder_path} is not a valid directory."
    
    subfolder_lengths_lookup = {}

    # Iterate through each subfolder in the root folder
    for subfolder in sorted(root_folder.iterdir()):
        if subfolder.is_dir():  # Ensure it's a directory
            # Use glob to find all .feather files and convert iterator to list to count
            feather_files = sorted(subfolder.glob('*.feather'))
            feather_file_lengths = [
                len(load_feather(feather_file)) for feather_file in feather_files
            ]
            subfolder_lengths_lookup[subfolder.name] = feather_file_lengths

    return subfolder_lengths_lookup

def save_counts_to_file(subfolder_count_lookup, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(subfolder_count_lookup, f, indent=4, sort_keys=True)

def main(root_folder_path : Path):
    subfolder_lengths_lookup = feather_file_lengths_in_subfolders(root_folder_path)
    

    # Construct output file name based on root folder name
    counts_output_file_name = f"{Path(args.root_folder_path).name}_counts.json"
    pc_lengths_output_file_name = f"{Path(args.root_folder_path).name}_pc_lengths.json"
    # Construct full output path
    counts_output_file_path = Path(args.root_folder_path).parent / counts_output_file_name
    pc_lengths_output_file_path = Path(args.root_folder_path).parent / pc_lengths_output_file_name
    save_json(counts_output_file_path, {k : len(v) for k, v in sorted(subfolder_lengths_lookup.items())}, indent=4)
    save_json(pc_lengths_output_file_path, subfolder_lengths_lookup, indent=4)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count .feather files in subfolders of a given directory.')
    parser.add_argument('root_folder_path', type=Path, help='Path to the root folder.')
    args = parser.parse_args()
    main(args.root_folder_path)
