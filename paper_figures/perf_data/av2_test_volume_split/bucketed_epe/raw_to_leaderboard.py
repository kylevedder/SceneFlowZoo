import json
from pathlib import Path
import numpy as np


def load_raw(raw_path: Path, is_supervised: bool) -> dict[str, float | int]:
    try:
        raw_datastruct = json.load(raw_path.open("r"))
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse {raw_path} as JSON")

    def _parse_str_tuple(str_tuple: str) -> tuple[float, float]:
        # Remove the parentheses and split the string
        split = str_tuple[1:-1].split(", ")
        # Convert the strings to floats
        return float(split[0].strip()), float(split[1].strip())

    # Convert the strings to floats
    float_datastruct = {key: _parse_str_tuple(value) for key, value in raw_datastruct.items()}

    # Calculate the mean dynamic EPE
    mean_dynamic = float(np.nanmean([value[1] for value in float_datastruct.values()]))
    mean_static = float(np.nanmean([value[0] for value in float_datastruct.values()]))

    output_dict = {
        "mean Dynamic": mean_dynamic,
        "mean Static": mean_static,
        "Is Supervised?": 1 if is_supervised else 0,
    }

    # Add the category performance
    for key, value in float_datastruct.items():
        if np.isfinite(value[0]):
            output_dict[key + " Static"] = value[0]
        if np.isfinite(value[1]):
            output_dict[key + " Dynamic"] = value[1]
    return output_dict


def save_json(data: dict[str, float | int], save_path: Path):
    with save_path.open("w") as file:
        json.dump([data], file, indent=4)


def process(raw_file: Path, save_file: Path, is_supervised: bool):
    raw_data = load_raw(raw_file, is_supervised)
    save_json(raw_data, save_file)


root_folder = Path()
# fmt: off
process(root_folder / "deflow_raw.json", root_folder / "deflow.json", is_supervised=True)
process(root_folder / "fastflow3d_raw.json", root_folder / "fastflow3d.json", is_supervised=True)
process(root_folder / "nsfp_raw.json", root_folder / "nsfp.json", is_supervised=False)
process(root_folder / "trackflow_raw.json", root_folder / "trackflow.json", is_supervised=False)
process(root_folder / "zeroflow_1x_raw.json", root_folder / "zeroflow_1x.json", is_supervised=False)
process(root_folder / "zeroflow_3x_raw.json", root_folder / "zeroflow_3x.json", is_supervised=False)
process(root_folder / "zeroflow_5x_raw.json", root_folder / "zeroflow_5x.json", is_supervised=False)
process(root_folder / "zeroflow_xl_3x_raw.json", root_folder / "zeroflow_xl_3x.json", is_supervised=False)
process(root_folder / "zeroflow_xl_5x_raw.json", root_folder / "zeroflow_xl_5x.json", is_supervised=False)
# fmt: on
