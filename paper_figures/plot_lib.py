import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import json
from bucketed_scene_flow_eval.datasets.argoverse2.av2_metacategories import BUCKETED_METACATAGORIES

linewidth = 0.5
minor_tick_color = (0.9, 0.9, 0.9)


class ThreewayEvalStats:

    def __init__(self, data_root_dir: Path, clean_name: str, data_name: str):
        self.name = clean_name
        data_file = data_root_dir / (data_name + ".json")
        assert data_file.exists(), f"Data file {data_file} does not exist"
        # Load the json
        with open(data_file, "r") as file:
            data_structure = json.load(file)
        BACKGROUND_KEY = "BACKGROUND"
        FOREGROUND_KEY = "FOREGROUND"

        # Ensure that the only keys are "BACKGROUND" and "FOREGROUND"
        assert len(data_structure.keys()) == 2, f"Expected 2 keys, got {len(data_structure.keys())}"
        assert BACKGROUND_KEY in data_structure, f"Expected key {BACKGROUND_KEY} not found"
        assert FOREGROUND_KEY in data_structure, f"Expected key {FOREGROUND_KEY} not found"

        background_tuple = self._parse_str_tuple(data_structure[BACKGROUND_KEY])
        foreground_tuple = self._parse_str_tuple(data_structure[FOREGROUND_KEY])

        self.background_static = background_tuple[0]
        self.foreground_static = foreground_tuple[0]
        self.foreground_dynamic = foreground_tuple[1]

        self.mean_threeway = (
            self.background_static + self.foreground_static + self.foreground_dynamic
        ) / 3

    def _parse_str_tuple(self, str_tuple: str) -> Tuple[float, float]:
        # Remove the parentheses and split the string
        split = str_tuple[1:-1].split(", ")
        # Convert the strings to floats
        return float(split[0].strip()), float(split[1].strip())

    def __lt__(self, other: "ThreewayEvalStats"):
        return self.mean_threeway < other.mean_threeway


class BucketedEvalStats:

    def __init__(self, data_root_dir: Path, clean_name: str, data_name: str):
        self.name = clean_name
        data_file = data_root_dir / (data_name + ".json")
        assert data_file.exists(), f"Data file {data_file} does not exist"
        self.data = self._load_data(data_file)

    def _convert_to_leaderboard_data(self, raw_datastruct: dict[str, str]):
        def _parse_str_tuple(str_tuple: str) -> Tuple[float, float]:
            # Remove the parentheses and split the string
            split = str_tuple[1:-1].split(", ")
            # Convert the strings to floats
            return float(split[0].strip()), float(split[1].strip())

        # Convert the strings to floats
        float_datastruct = {key: _parse_str_tuple(value) for key, value in raw_datastruct.items()}

        # Calculate the mean dynamic EPE
        mean_dynamic = np.nanmean([value[1] for value in float_datastruct.values()])
        mean_static = np.nanmean([value[0] for value in float_datastruct.values()])

        output_dict = {
            "mean Dynamic": mean_dynamic,
            "mean Static": mean_static,
            "Is Supervised?": 1 if "supervised" in raw_datastruct else 0,
        }

    def _load_data(self, path: Path) -> dict[str, float]:
        # load the file into a string
        with open(path, "r") as file:
            file_content = file.read()
        # replace single with double quotes to be JSON compliant
        file_content = file_content.replace("'", '"')
        # load the string as a JSON object
        data_structure = json.loads(file_content)
        return data_structure[0]

    def get_dynamic_metacatagory_performance(self) -> Dict[str, float]:
        # All entries that contain "Dynamic" in the key, except for "mean Dynamic"
        return {
            key: value
            for key, value in self.data.items()
            if "Dynamic" in key and key != "mean Dynamic"
        }

    def mean_dynamic(self) -> float:
        return self.data["mean Dynamic"]

    def mean_static(self) -> float:
        return self.data["mean Static"]

    def is_supervised(self) -> bool:
        is_supervised_int = self.data["Is Supervised?"]
        return bool(is_supervised_int)

    # Comparator to sort by increasing mean Dynamic EPE
    def __lt__(self, other):
        return self.mean_dynamic() < other.mean_dynamic()


def set_font(size):
    matplotlib.rcParams.update(
        {  # Use mathtext, not LaTeX
            "text.usetex": False,
            # Use the Computer modern font
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "font.size": size,
            "mathtext.fontset": "cm",
            # Use ASCII minus
            "axes.unicode_minus": False,
        }
    )


def centered_barchart_offset(elem_idx: int, total_elems: int, bar_width: float) -> float:
    """
    Calculate the x offset for a bar in a barchart so that the bars are centered around zero.

    Handle both odd and even total_elems cases.
    """
    if total_elems % 2 == 1:  # Odd number of elements
        # Middle index
        middle_idx = total_elems // 2
        # Calculate offset
        offset = (elem_idx - middle_idx) * bar_width
    else:  # Even number of elements
        # Calculate half of the space occupied by all bars
        half_total_width = (total_elems * bar_width) / 2
        # Offset for the left side of the middle two bars
        middle_left_offset = half_total_width - (bar_width / 2)
        # Calculate offset
        offset = ((elem_idx + 0.5) * bar_width) - middle_left_offset

    return offset


def color_map(rev: bool = False):
    # return 'gist_earth'
    if rev:
        return "magma_r"
    return "magma"


def color(count, total_elements, intensity=1.3):
    start = 0.2
    stop = 0.7

    colormap = matplotlib.cm.get_cmap(color_map())
    cm_subsection = np.linspace(start, stop, total_elements)
    # color = [matplotlib.cm.gist_earth(x) for x in cm_subsection][count]
    color = [colormap(x) for x in cm_subsection][count]
    # Scale the color by intensity while leaving the 4th channel (alpha) unchanged
    return [min(x * intensity, 1) for x in color[:3]] + [color[3]]


def color2d(count_x, count_y, total_elements_x, total_elements_y):
    # Select the actual color, then scale along the intensity axis
    start = 1.7
    stop = 1
    intensity_scale = np.linspace(start, stop, total_elements_y)
    intensity = intensity_scale[count_y]
    return color(count_x, total_elements_x, intensity)


def grid(minor=True, axis="both"):
    plt.grid(linewidth=linewidth / 2, axis=axis)
    if minor:
        plt.grid(
            which="minor",
            color=minor_tick_color,
            linestyle="--",
            alpha=0.7,
            clip_on=True,
            linewidth=linewidth / 4,
            zorder=0,
        )


def savefig(save_folder: Path, name: str, pad: float = 0):
    save_folder = Path(save_folder)
    for ext in ["pdf", "png"]:
        outfile = save_folder / f"{name}.{ext}"
        print("Saving", outfile)
        plt.savefig(outfile, bbox_inches="tight", pad_inches=pad)
    plt.clf()


def savetable(save_folder: Path, name: str, content: List[List[Any]]):
    outfile = save_folder / f"{name}.txt"

    def fmt(e):
        if (
            type(e) == float
            or type(e) == np.float64
            or type(e) == np.float32
            or type(e) == np.float16
        ):
            return f"{e:.3f}"
        return str(e)

    print("Saving", outfile)
    with open(outfile, "w") as f:

        assert type(content) == list, "Table must be a list of rows"
        for row in content:
            assert type(row) == list, "Table rows must be lists"
            f.write(" & ".join([fmt(e) for e in row]) + "\\\\\n")
