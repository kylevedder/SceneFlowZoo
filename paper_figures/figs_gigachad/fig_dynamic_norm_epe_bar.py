from paper_figures.plot_lib import BucketedEvalStats, color, savefig, centered_barchart_offset
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path


def plot_dynamic_norm_epe_bar_black(eval_stats: List[BucketedEvalStats], save_folder: Path):
    # Make bar chart of mean average EPE dynamic
    names = [eval_stat.name for eval_stat in eval_stats]
    normalized_epes = [eval_stat.mean_dynamic() for eval_stat in eval_stats]
    colors = [
        color(idx, len(eval_stats)) if not eval_stat.is_supervised() else "black"
        for idx, eval_stat in enumerate(eval_stats)
    ]
    hatches = [None if eval_stat.is_supervised() else None for eval_stat in eval_stats]

    ##### BarH head to head
    fig_size = 5.5
    plt.gcf().set_size_inches(fig_size, fig_size / 1.6)
    plt.barh(names, normalized_epes, color=colors, hatch=hatches, edgecolor="black")
    # Add text labels for each bar
    for name, normalized_epe in zip(names, normalized_epes):
        plt.text(normalized_epe + 0.01, name, f"{normalized_epe:.4f}", va="center", color="black")

    right_max = 0.5
    # plt.xlim(left=0, right=right_max)
    plt.xlabel("mean Dynamic Normalized EPE")

    # Remove top and right axes
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_yticklabels():
        if "(Ours)" in label.get_text():
            label.set_fontweight("bold")
            print(f"bolding {label.get_text()}")
    savefig(save_folder, "mean_average_bucketed_epe_black")


def plot_dynamic_norm_epe_bar_black_fixed_width_bar(
    eval_stats: List[BucketedEvalStats], save_folder: Path
):

    fig_size = 5.5
    bar_width = 0.12
    bar_gap = 0.03

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
            normalized_epe + 0.01,
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

    plt.gcf().set_size_inches(fig_size, len(eval_stats) * (bar_width + bar_gap) + 0.5)
    plt.xlabel("mean Dynamic Normalized EPE")

    # Remove top and right axes
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_yticklabels():
        if "(Ours)" in label.get_text():
            label.set_fontweight("bold")
            print(f"bolding {label.get_text()}")
    savefig(save_folder, "mean_average_bucketed_epe_black")


def plot_dynamic_norm_epe_bar(eval_stats: List[BucketedEvalStats], save_folder: Path):
    # Make bar chart of mean average EPE dynamic
    names = [eval_stat.name for eval_stat in eval_stats]
    normalized_epes = [eval_stat.mean_dynamic() for eval_stat in eval_stats]
    colors = [color(idx, len(eval_stats)) for idx, _ in enumerate(eval_stats)]
    hatches = ["/" if eval_stat.is_supervised() else None for eval_stat in eval_stats]

    ##### BarH head to head
    fig_size = 5.5
    plt.gcf().set_size_inches(fig_size, fig_size / 1.6)
    plt.barh(names, normalized_epes, color=colors, hatch=hatches, edgecolor="black")
    # Add text labels for each bar
    for name, normalized_epe in zip(names, normalized_epes):
        plt.text(normalized_epe + 0.01, name, f"{normalized_epe:.4f}", va="center", color="black")

    right_max = 0.5
    # plt.xlim(left=0, right=right_max)
    plt.xlabel("mean Dynamic Normalized EPE")

    # Remove top and right axes
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_yticklabels():
        if "(Ours)" in label.get_text():
            label.set_fontweight("bold")
            print(f"bolding {label.get_text()}")
    savefig(save_folder, "mean_average_bucketed_epe")
