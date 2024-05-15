from paper_figures.plot_lib import BucketedEvalStats, color, savefig
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path


def plot_dynamic_norm_epe_bar(eval_stats: List[BucketedEvalStats], save_folder: Path):
    # Make bar chart of mean average EPE dynamic
    names = [eval_stat.name for eval_stat in eval_stats]
    normalized_epes = [eval_stat.mean_dynamic() for eval_stat in eval_stats]
    colors = [color(idx, len(eval_stats)) for idx, _ in enumerate(eval_stats)]
    hatches = ["/" if eval_stat.is_supervised() else None for eval_stat in eval_stats]

    ##### BarH head to head
    fig_size = 5
    plt.gcf().set_size_inches(fig_size, fig_size / 1.6)
    plt.barh(names, normalized_epes, color=colors, hatch=hatches, edgecolor="black")
    # Add text labels for each bar
    for name, normalized_epe in zip(names, normalized_epes):
        plt.text(normalized_epe + 0.01, name, f"{normalized_epe:.4f}", va="center", color="black")

    right_max = 1.0
    plt.xlim(left=0, right=right_max)
    plt.xlabel("mean Dynamic Normalized EPE")

    # Remove top and right axes
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(save_folder, "mean_average_bucketed_epe")
