import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


class PerfData:
    def __init__(self, name: str, data: dict[str, tuple[float, float]]):
        self.name = name
        self.data = data

    @staticmethod
    def from_results_folder(results_folder: Path) -> "PerfData":

        def parse_tuple_string(val: str) -> tuple[float, float]:
            static_epe, dynamic_normalized_epe = val.strip()[1:-1].split(",")
            static_epe = static_epe.strip()
            dynamic_normalized_epe = dynamic_normalized_epe.strip()
            return float(static_epe), float(dynamic_normalized_epe)

        json_file = results_folder / "per_class_results_35.json"
        with json_file.open("r") as f:
            json_data: dict[str, str] = json.load(f)
        return PerfData(
            results_folder.name, {k: parse_tuple_string(v) for k, v in json_data.items()}
        )

    @property
    def mean_dynamic(self):
        return np.nanmean([v[1] for v in self.data.values()])

    @property
    def mean_static(self):
        return np.nanmean([v[0] for v in self.data.values()])


def process_results_folders(results_folders: list[Path]) -> list[PerfData]:

    perf_data_lst = [
        PerfData.from_results_folder(results_folder) for results_folder in results_folders
    ]

    return sorted(perf_data_lst, key=lambda x: x.mean_dynamic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path, help="Path to the root results folder")
    parser.add_argument("--full_table", action="store_true", help="Print the full table")
    parser.add_argument("--full_dynamic", action="store_true", help="Print the dynamic entries")
    parser.add_argument("--full_static", action="store_true", help="Print the static entries")
    args = parser.parse_args()

    perf_datas = process_results_folders(list(args.results_root.glob("*/")))

    # Construct the DataFrame
    results_dict = {
        "Name": [item.name for item in perf_datas],
        "Mean Static": [item.mean_static for item in perf_datas],
        "Mean Dynamic": [item.mean_dynamic for item in perf_datas],
    }

    print_static = args.full_table or args.full_static
    print_dynamic = args.full_table or args.full_dynamic

    # Expand nested data for each self.data key
    for perf_data in perf_datas:
        for key, value in sorted(perf_data.data.items()):
            if print_static:
                results_dict[f"Sta({key})"] = value[0]
            if print_dynamic:
                results_dict[f"Dyn({key})"] = value[1]

    df = pd.DataFrame(results_dict)

    # Print the table
    print(df.to_string())
