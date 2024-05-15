from paper_figures.plot_lib import BucketedEvalStats, ThreewayEvalStats, color, savefig
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
import copy


def _get_fig_size():
    return 3.8


# fig_size = 2.85


def plot_threeway_epe_bar(
    eval_stats: List[tuple[BucketedEvalStats, ThreewayEvalStats]], save_folder: Path
):
    fig_size = _get_fig_size()
    eval_stats = copy.deepcopy(eval_stats)
    eval_stats.sort(key=lambda x: x[1].mean_threeway, reverse=True)

    # Make bar chart of mean average EPE dynamic
    names = [eval_stat[0].name for eval_stat in eval_stats]
    normalized_epes = [eval_stat[1].mean_threeway for eval_stat in eval_stats]
    colors = [color(idx, len(eval_stats)) for idx, _ in enumerate(eval_stats)]
    hatches = ["/" if eval_stat[0].is_supervised() else None for eval_stat in eval_stats]

    ##### BarH head to head

    plt.gcf().set_size_inches(fig_size / 2, fig_size / 1.6)
    plt.barh(names, normalized_epes, color=colors, hatch=hatches, edgecolor="black")
    right_max = max(normalized_epes) * 1.10
    # Add text labels for each bar
    for name, normalized_epe in zip(names, normalized_epes):
        plt.text(
            normalized_epe + right_max * 0.01,
            name,
            f"{normalized_epe:.4f}",
            va="center",
            color="black",
        )

    plt.xlim(left=0, right=right_max)
    # plt.xlabel("Threeway EPE")
    plt.xlabel("Average EPE (m)")

    # Remove top and right axes
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(save_folder, "threeway_epe")


def plot_threeway_dynamic_epe_bar(
    eval_stats: List[tuple[BucketedEvalStats, ThreewayEvalStats]], save_folder: Path
):
    fig_size = _get_fig_size()
    eval_stats = copy.deepcopy(eval_stats)
    eval_stats.sort(key=lambda x: x[1].mean_threeway, reverse=True)
    eval_stats_and_color = [
        (eval_stat[0], eval_stat[1], color_idx) for color_idx, eval_stat in enumerate(eval_stats)
    ]
    eval_stats_and_color.sort(key=lambda x: x[1].foreground_dynamic, reverse=True)

    # Make bar chart of mean average EPE dynamic
    names = [eval_stat[0].name for eval_stat in eval_stats_and_color]
    normalized_epes = [eval_stat[1].foreground_dynamic for eval_stat in eval_stats_and_color]
    colors = [color(eval_stat[2], len(eval_stats_and_color)) for eval_stat in eval_stats_and_color]
    hatches = ["/" if eval_stat[0].is_supervised() else None for eval_stat in eval_stats_and_color]

    ##### BarH head to head

    plt.gcf().set_size_inches(fig_size / 2, fig_size / 1.6)
    plt.barh(names, normalized_epes, color=colors, hatch=hatches, edgecolor="black")
    right_max = max(normalized_epes) * 1.10
    # Add text labels for each bar
    for name, normalized_epe in zip(names, normalized_epes):
        plt.text(
            normalized_epe + right_max * 0.01,
            name,
            f"{normalized_epe:.4f}",
            va="center",
            color="black",
        )

    plt.xlim(left=0, right=right_max)
    plt.xlabel("Average EPE (m)")

    # Remove top and right axes
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(save_folder, "threeway_epe_dynamic")
