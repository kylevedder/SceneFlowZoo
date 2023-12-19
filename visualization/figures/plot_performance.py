from pathlib import Path
from loader_utils import *
from visualization.figures.plot_lib import *
import argparse
import json

# Get path to methods from command line
parser = argparse.ArgumentParser()
parser.add_argument('output_folder', type=Path)
args = parser.parse_args()

assert args.output_folder.exists(
), f"Output folder {args.output_folder} does not exist"

save_folder = args.output_folder
save_folder.mkdir(exist_ok=True, parents=True)
nsfp_3x = Path("/tmp/frame_results/bucketed_epe/nsfp_distillation_3x/")
nsfp_1x = Path("/tmp/frame_results/bucketed_epe/nsfp_distillation_1x/")
supervised = Path("/tmp/frame_results/bucketed_epe/supervised/")


def load_class_results(root_path: Path, distance=35):
    load_path = root_path / f"per_class_results_{distance}.json"
    with open(load_path) as f:
        data = json.load(f)

    # Convert dictionary of str, str to dictionary of str, Tuple[float, float]
    data = {k: eval(v.replace("-", "np.NaN")) for k, v in data.items()}
    return data


#################

# Load data
nsfp_3x_data = load_class_results(nsfp_3x)
nsfp_1x_data = load_class_results(nsfp_1x)
supervised_data = load_class_results(supervised)


def average_data(data):
    statics, dynamics = zip(*data.values())
    return np.nanmean(statics), np.nanmean(dynamics)


print("NSFP 3x")
print(average_data(nsfp_3x_data))
print("NSFP 1x")
print(average_data(nsfp_1x_data))
print("Supervised")
print(average_data(supervised_data))
