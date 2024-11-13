from paper_figures.plot_lib import (
    BucketedEvalStats,
    ThreewayEvalStats,
    color,
    savefig,
    centered_barchart_offset,
)
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
import copy


def plot_ablation_barchart(eval_stats: List[BucketedEvalStats], save_folder: Path):

    fig_size = 5.5
    bar_width = 0.1
    bar_gap = 0.03

    max_normalized_epe = max([eval_stat.mean_dynamic() for eval_stat in eval_stats])
    for idx, eval_stat in enumerate(eval_stats):
        y_position = centered_barchart_offset(idx, len(eval_stats), bar_width, bar_gap)
        normalized_epe = eval_stat.mean_dynamic()
        bar_color = color(idx, len(eval_stats)) if not eval_stat.is_supervised() else "black"
        plt.barh(
            y=y_position,
            width=normalized_epe,
            height=bar_width,
            color=bar_color,
            hatch=None,
            edgecolor="black",
        )
        plt.text(
            normalized_epe + 0.01 * (max_normalized_epe / 0.75),
            y_position,
            f"{normalized_epe:.4f}",
            va="center",
            color="black",
        )

    # Set Y axis labels
    plt.yticks(
        [
            centered_barchart_offset(idx, len(eval_stats), bar_width, bar_gap)
            for idx in range(len(eval_stats))
        ],
        [eval_stat.name for eval_stat in eval_stats],
    )

    plt.gcf().set_size_inches(fig_size, len(eval_stats) * (bar_width + bar_gap) + 0.3)
    plt.xlabel("mean Dynamic Normalized EPE")

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_folder.mkdir(exist_ok=True, parents=True)
    savefig(save_folder, "ablation_barchart")
