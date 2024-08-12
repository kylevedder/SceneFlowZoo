import argparse
import zipfile
import re
from pathlib import Path


def get_latest_and_best_checkpoint(job_dir: Path) -> tuple[Path | None, Path | None]:

    def has_valid_dataset_idx_dir(timestamp_dir: Path) -> bool:
        has_dataset_dir = any(
            subdir.name.startswith("dataset_idx_") for subdir in timestamp_dir.iterdir()
        )
        return has_dataset_dir

    # Find all subdirectories matching the timestamp pattern
    timestamp_dirs = sorted(
        e for e in job_dir.glob("*") if e.is_dir() and has_valid_dataset_idx_dir(e)
    )

    assert timestamp_dirs, f"No timestamp directories found in {job_dir}"

    # Sort timestamp directories by name (latest first)
    latest_timestamp_dir = max(timestamp_dirs, key=lambda x: x.name)

    # Find all .pth files in the latest timestamp directory
    pth_files = list(latest_timestamp_dir.glob("dataset_idx_*/epoch_*.pth"))

    assert pth_files, f"No .pth files found in {latest_timestamp_dir}"

    def get_epoch_number(file_path: Path) -> int:
        match = re.search(r"epoch_(\d+)_checkpoint", file_path.name)
        assert match is not None, f"Failed to extract epoch number from {file_path.name}"
        return int(match.group(1))

    latest_checkpoint = max(pth_files, key=get_epoch_number)
    best_checkpoint = latest_timestamp_dir / "dataset_idx_0000000000" / "best_weights.pth"

    if not latest_checkpoint.exists():
        latest_checkpoint = None
    if not best_checkpoint.exists():
        best_checkpoint = None

    return latest_checkpoint, best_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Zip the latest checkpoint files from all job directories."
    )
    parser.add_argument("input_path", type=Path, help="Input directory path")
    parser.add_argument("-o", "--output_zip", type=Path, help="Output zip file name (optional)")
    args = parser.parse_args()

    if args.output_zip is None:
        args.output_zip = args.input_path.with_suffix(".zip")

    # Find all job directories
    job_dirs = list(args.input_path.glob("job_*"))
    assert len(job_dirs) > 0, f"No job directories found in {args.input_path}"

    try:
        # Remove the zip file if it already exists
        if args.output_zip.exists():
            args.output_zip.unlink()
            print(f"Removed existing zip file: {args.output_zip}")

        with zipfile.ZipFile(args.output_zip, "w") as zipf:
            for job_dir in job_dirs:
                latest_checkpoint, best_checkpoint = get_latest_and_best_checkpoint(job_dir)

                def add_checkpoint_if_exists(checkpoint: Path | None):
                    if not checkpoint:
                        return

                    zip_path = Path(job_dir.name) / checkpoint.name
                    zipf.write(checkpoint, zip_path)
                    print(f"Added {zip_path} to {args.output_zip}")

                add_checkpoint_if_exists(latest_checkpoint)
                add_checkpoint_if_exists(best_checkpoint)

        print(f"Successfully created {args.output_zip}")
    except Exception as e:
        print(f"Error creating zip file: {e}")


if __name__ == "__main__":
    main()
