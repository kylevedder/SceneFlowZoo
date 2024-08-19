from paper_figures.plot_lib import BucketedEvalStats, ThreewayEvalStats, color, savefig
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
import copy

fig_size = 5.5


def plot_ablation_barchart(eval_stats: List[BucketedEvalStats], save_folder: Path):
    plt.gcf().set_size_inches(fig_size, fig_size / 1.6)

    names = [eval_stat.name for eval_stat in eval_stats]
    dynamic_epes = [eval_stat.mean_dynamic() for eval_stat in eval_stats]
    colors = [color(idx, len(eval_stats)) for idx, _ in enumerate(eval_stats)]
    hatches = ["/" if eval_stat.is_supervised() else None for eval_stat in eval_stats]
    plt.barh(names, dynamic_epes, color=colors, hatch=hatches, edgecolor="black")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_folder.mkdir(exist_ok=True, parents=True)
    savefig(save_folder, "ablation_barchart")
