from pathlib import Path
from loader_utils import *
from visualization.figures.plot_lib import *
import argparse
import json
import numpy as np
from loader_utils import load_json
from dataclasses import dataclass
from bucketed_scene_flow_eval.datasets.argoverse2.av2_metacategories import METACATAGORIES

# Get path to methods from command line
parser = argparse.ArgumentParser()
parser.add_argument('output_folder', type=Path)
args = parser.parse_args()

assert args.output_folder.exists(
), f"Output folder {args.output_folder} does not exist"

save_folder = args.output_folder
save_folder.mkdir(exist_ok=True, parents=True)


class EvalStats():

    def __init__(self, name: str, subfolder: str, is_supervised: bool = False):
        self.name = name
        self.data_folder = data_root_dir / subfolder
        assert self.data_folder.exists(
        ), f"Data folder {self.data_folder} does not exist"
        self.is_supervised = is_supervised

    def class_averages(self, distance=35) -> Dict[str, Tuple[float, float]]:
        load_path = self.data_folder / f"per_class_results_{distance}.json"

        data = load_json(load_path)

        # Convert dictionary of str, str to dictionary of str, Tuple[float, float]
        data = {
            k: v.replace("-", "np.NaN").replace("nan", "np.NaN")
            for k, v in data.items()
        }
        try:
            data = {k: eval(v) for k, v in data.items()}
        except Exception as e:
            print(data)
            raise e
        return data

    def mean_average(self, distance=35) -> Tuple[float, float]:
        class_averages = self.class_averages(distance)
        statics, dynamics = zip(*class_averages.values())
        return np.nanmean(statics), np.nanmean(dynamics)



#################

# Load data

data_root_dir = Path("/tmp/frame_results/bucketed_epe/")

eval_stats = [
    EvalStats("FastFlow3D", "supervised", is_supervised=True),
    EvalStats("ZeroFlow 1x", "nsfp_distillation_1x"),
    EvalStats("ZeroFlow 3x", "nsfp_distillation_3x"),
    EvalStats("ZeroFlow 1x XL", "nsfp_distillation_xl_1x"),
    EvalStats("ZeroFlow 3x XL", "nsfp_distillation_xl_3x"),
]

eval_stats.reverse()

# Make bar chart of mean average EPE dynamic
names = [eval_stat.name for eval_stat in eval_stats]
normalized_epes = [eval_stat.mean_average()[1] for eval_stat in eval_stats]
colors = [color(idx, len(eval_stats)) for idx, _ in enumerate(eval_stats)]
hatches = [
    '/' if eval_stat.is_supervised else None for eval_stat in eval_stats
]

##### BarH head to head

plt.gcf().set_size_inches(6.5, 6.5 / 1.6)
plt.barh(names,
         normalized_epes,
         color=colors,
         hatch=hatches,
         edgecolor='black')
# Add text labels for each bar
for name, normalized_epe in zip(names, normalized_epes):
    plt.text(normalized_epe + 0.01,
             name,
             f"{normalized_epe:.3f}",
             va='center',
             color='black')

right_max = 1.0
plt.xlim(left=0, right=right_max)
plt.xlabel("mean Dynamic Normalized EPE")

# Remove top and right axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
savefig(save_folder, "mean_average_bucketed_epe")
