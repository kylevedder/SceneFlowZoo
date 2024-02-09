import argparse
import json
from pathlib import Path
import zipfile
import tqdm

def load_validation_data(validation_file_path: Path):
    with validation_file_path.open('r') as file:
        return json.load(file)

def validate_and_archive_feather_files(root_folder_path: Path, validation_data, archive_path: Path):
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for subfolder in tqdm.tqdm(sorted(root_folder_path.iterdir()), desc="Validating and archiving folders"):
            if subfolder.is_dir():  # Ensure it's a directory
                expected_count = validation_data.get(subfolder.name)
                feather_files = sorted(subfolder.glob('*.feather'))
                actual_count = len(feather_files)
                
                if expected_count is not None:
                    if actual_count != expected_count:
                        raise ValueError(f"Validation failed for '{subfolder.name}': expected {expected_count}, found {actual_count}")
                else:
                    raise ValueError(f"No validation data for '{subfolder.name}'. Found {actual_count} items.")
                
                # Add every 5th .feather file to the zip, preserving the subfolder structure
                for i, feather_file in enumerate(feather_files):
                    if i % 5 == 0:  # Add every 5th file
                        # Calculate path relative to the root folder to preserve structure
                        relative_path = feather_file.relative_to(root_folder_path)
                        save_relative_path = relative_path.parent / f"{i:010d}.feather"
                        zipf.write(feather_file, arcname=str(save_relative_path))
        print(f"Archive created at '{archive_path}'")

def main():
    parser = argparse.ArgumentParser(description="Validate the number of .feather files in subfolders against a validation JSON and archive every 5th file into a single zip, preserving the subfolder structure.")
    parser.add_argument('root_folder_path', type=Path, help="Path to the root folder.")
    parser.add_argument('--validation_json', type=Path, default="data_prep_scripts/argo/av2_test_sizes.json", help="Path to the validation JSON file.")
    parser.add_argument('--archive_path', type=Path, help="Path to the output archive zip file.", default=None)
    
    args = parser.parse_args()
    
    if args.archive_path is None:
        # Default archive path: root folder name + '_av2_2024_sf_submission.zip'
        default_archive_name = f"{args.root_folder_path.name}_av2_2024_sf_submission.zip"
        args.archive_path = args.root_folder_path.parent / default_archive_name
    
    # Load validation data
    validation_data = load_validation_data(args.validation_json)

    # Remove archive if it already exists
    if args.archive_path.exists():
        args.archive_path.unlink()
    
    # Validate folder counts and archive every 5th feather file
    validate_and_archive_feather_files(args.root_folder_path, validation_data, args.archive_path)

if __name__ == "__main__":
    main()
