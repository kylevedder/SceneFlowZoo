from paper_figures.plot_lib import (
    BucketedEvalStats,
    color,
    savefig,
    centered_barchart_offset,
)
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


def _get_hatching(eval_stat: BucketedEvalStats) -> Optional[str]:
    return "/" if eval_stat.is_supervised() else None


def _clean_name(name: str):
    return name.replace("_", " ").replace("Dynamic", "").strip()


def _get_value(eval_stat: BucketedEvalStats, category_name: str) -> float:
    return eval_stat.get_dynamic_metacatagory_performance()[category_name]


def plot_per_metacatagory_bar(eval_stats: List[BucketedEvalStats], save_folder: Path):
    fig_size = 5
    eval_stats = sorted(eval_stats)

    category_names = sorted(eval_stats[0].get_dynamic_metacatagory_performance().keys())

    for category_idx, category_name in enumerate(category_names):
        plt.gcf().set_size_inches(fig_size / 2, fig_size / 1.6 / 2)

        bar_width = 0.1  # Define the width of each bar

        for idx, eval_stat in enumerate(eval_stats):
            value = _get_value(eval_stat, category_name)
            y_position = centered_barchart_offset(idx, len(eval_stats), bar_width)

            plt.bar(
                y_position,
                value,
                width=bar_width,
                color=color(len(eval_stats) - idx - 1, len(eval_stats)),
                hatch=_get_hatching(eval_stat),
                edgecolor="black",
                label=eval_stat.name,
            )

        # Add axis labels for each bar based on metacatagory
        plt.xticks([], [])
        plt.ylim(0, 1)
        print
        if "CAR" in category_name or "OTHER_VEHICLES" in category_name:
            # Plot the name of the method vertically above the bar, rotated 90 degrees
            for idx, eval_stat in enumerate(eval_stats):
                value = _get_value(eval_stat, category_name)
                plt.text(
                    centered_barchart_offset(idx, len(eval_stats), bar_width),
                    value + 0.03,
                    eval_stat.name,
                    ha="center",
                    va="baseline",
                    rotation=90,
                )

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if not ("WHEELED_VRU" in category_name or "OTHER_VEHICLES" in category_name):
            plt.ylabel("Dynamic Normalized EPE")

        savefig(save_folder, f"per_metacatagory_bar_{_clean_name(category_name)}")
