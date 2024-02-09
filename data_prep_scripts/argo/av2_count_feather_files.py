import argparse
import json
from pathlib import Path

def count_feather_files_in_subfolders(root_folder_path):
    # Convert the input path to a Path object for easy handling
    root_folder = Path(root_folder_path)
    
    # Check if the provided root folder path exists and is a directory
    if not root_folder.is_dir():
        print(f"The path {root_folder_path} is not a valid directory.")
        return
    
    subfolder_count_lookup = {}

    # Iterate through each subfolder in the root folder
    for subfolder in sorted(root_folder.iterdir()):
        if subfolder.is_dir():  # Ensure it's a directory
            # Use glob to find all .feather files and convert iterator to list to count
            feather_files_count = len(list(subfolder.glob('*.feather')))
            subfolder_count_lookup[subfolder.name] = feather_files_count

    return subfolder_count_lookup

def save_counts_to_file(subfolder_count_lookup, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(subfolder_count_lookup, f, indent=4, sort_keys=True)

def main():
    parser = argparse.ArgumentParser(description='Count .feather files in subfolders of a given directory.')
    parser.add_argument('root_folder_path', type=Path, help='Path to the root folder.')
    
    args = parser.parse_args()
    subfolder_count_lookup = count_feather_files_in_subfolders(args.root_folder_path)
    
    if subfolder_count_lookup is not None:
        # Construct output file name based on root folder name
        output_file_name = f"{Path(args.root_folder_path).name}_counts.json"
        # Construct full output path
        output_file_path = Path(args.root_folder_path).parent / output_file_name
        save_counts_to_file(subfolder_count_lookup, output_file_path)
        print(f"Counts saved to {output_file_path}")

if __name__ == "__main__":
    main()
