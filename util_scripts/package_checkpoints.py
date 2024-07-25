import argparse
import zipfile
import re
from pathlib import Path


def get_latest_checkpoint(base_path: Path) -> Path | None:
    subdirs = list(base_path.glob("job_*/*"))

    if not subdirs:
        return None

    latest_subdir = max(subdirs, key=lambda x: x.name)
    pth_files = list(latest_subdir.glob("dataset_idx_*/*.pth"))

    if not pth_files:
        return None

    def get_epoch_number(file_path: Path) -> int:
        match = re.search(r"epoch_(\d+)_checkpoint", file_path.name)
        assert match is not None, f"Failed to extract epoch number from {file_path.name}"
        return int(match.group(1))

    return max(pth_files, key=get_epoch_number)


def main():
    parser = argparse.ArgumentParser(description="Zip the latest checkpoint file.")
    parser.add_argument("input_path", type=Path, help="Input directory path")
    parser.add_argument("-o", "--output_zip", type=Path, help="Output zip file name (optional)")
    args = parser.parse_args()

    if args.output_zip is None:
        args.output_zip = args.input_path.with_suffix(".zip")

    latest_checkpoint = get_latest_checkpoint(args.input_path)

    if latest_checkpoint is None:
        print(f"No .pth files found in {args.input_path}")
        return

    try:
        # Remove the zip file if it already exists
        if args.output_zip.exists():
            args.output_zip.unlink()
            print(f"Removed existing zip file: {args.output_zip}")
        with zipfile.ZipFile(args.output_zip, "w") as zipf:
            zipf.write(latest_checkpoint, latest_checkpoint.name)
        print(f"Successfully added {latest_checkpoint.name} to {args.output_zip}")
    except Exception as e:
        print(f"Error creating zip file: {e}")


if __name__ == "__main__":
    main()
