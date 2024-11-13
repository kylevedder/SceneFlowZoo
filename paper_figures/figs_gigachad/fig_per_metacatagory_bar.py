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
    dynamic_performances = eval_stat.get_dynamic_metacatagory_performance()
    assert (
        category_name in dynamic_performances
    ), f"Category {category_name} not found in {eval_stat.name}'s keys: {dynamic_performances.keys()}"
    return dynamic_performances[category_name]


def _get_dynamic_keys(eval_stats: List[BucketedEvalStats]) -> list[str]:
    intersected_name_set = set(eval_stats[0].get_dynamic_metacatagory_performance().keys())
    for eval_stat in eval_stats:
        # Intersect with the set of keys of the current eval_stat
        current_set = set(eval_stat.get_dynamic_metacatagory_performance().keys())
        intersected_name_set = intersected_name_set.intersection(current_set)

    return sorted(intersected_name_set)


def plot_per_metacatagory_bar_av2(eval_stats: List[BucketedEvalStats], save_folder: Path):
    fig_size = 5.5
    eval_stats = sorted(eval_stats)

    category_names = _get_dynamic_keys(eval_stats)

    for category_idx, category_name in enumerate(category_names):
        plt.gcf().set_size_inches(fig_size / 2, fig_size / 1.6 / 2)

        bar_width = 1  # Define the width of each bar
        bar_gap = 0.3  # Define the gap between each bar
        for idx, eval_stat in enumerate(eval_stats):
            value = _get_value(eval_stat, category_name)
            y_position = centered_barchart_offset(idx, len(eval_stats), bar_width, bar_gap)

            bar_color = (
                color(len(eval_stats) - idx - 1, len(eval_stats))
                if not eval_stat.is_supervised()
                else "black"
            )

            plt.bar(
                y_position,
                value,
                width=bar_width,
                color=bar_color,
                edgecolor="black",
                label=eval_stat.name,
            )

        # Add axis labels for each bar based on metacatagory
        plt.xticks([], [])
        plt.ylim(0, 1)

        # Add the numerical value to the plot at the bottom of the bar
        for idx, eval_stat in enumerate(eval_stats):
            y_position = centered_barchart_offset(idx, len(eval_stats), bar_width, bar_gap)
            value = _get_value(eval_stat, category_name)
            plt.text(
                y_position,
                value - 0.07,
                f"{value:.4f}",
                ha="center",
                va="baseline",
                rotation=90,
                color="black" if not eval_stat.is_supervised() else "white",
                # set the font size to 2
                fontsize=2,
            )

        # Add the category name to the plot
        if "CAR" in category_name or "OTHER_VEHICLES" in category_name:
            # Plot the name of the method vertically above the bar, rotated 90 degrees
            for idx, eval_stat in enumerate(eval_stats):
                y_position = centered_barchart_offset(idx, len(eval_stats), bar_width, bar_gap)
                value = _get_value(eval_stat, category_name)
                plt.text(
                    y_position,
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

        savefig(save_folder, f"per_metacatagory_bar_{_clean_name(category_name).replace(' ', '_')}")


def plot_per_metacatagory_bar_waymo(eval_stats: List[BucketedEvalStats], save_folder: Path):
    fig_size = 5.5
    eval_stats = sorted(eval_stats)

    category_names = _get_dynamic_keys(eval_stats)

    for category_idx, category_name in enumerate(category_names):
        plt.gcf().set_size_inches(fig_size / 3, fig_size / 1.6 / 2.8)

        bar_width = 1  # Define the width of each bar
        bar_gap = 0.3  # Define the gap between each bar
        for idx, eval_stat in enumerate(eval_stats):
            value = _get_value(eval_stat, category_name)
            y_position = centered_barchart_offset(idx, len(eval_stats), bar_width, bar_gap)

            bar_color = (
                color(len(eval_stats) - idx - 1, len(eval_stats))
                if not eval_stat.is_supervised()
                else "black"
            )

            plt.bar(
                y_position,
                value,
                width=bar_width,
                color=bar_color,
                edgecolor="black",
                label=eval_stat.name,
            )

        # Add axis labels for each bar based on metacatagory
        plt.xticks([], [])
        plt.ylim(0, 1)

        # Add the numerical value to the plot at the bottom of the bar
        for idx, eval_stat in enumerate(eval_stats):
            y_position = centered_barchart_offset(idx, len(eval_stats), bar_width, bar_gap)
            value = _get_value(eval_stat, category_name)
            rotation = 90
            y_offset = -0.0
            if "VEHICLE" in category_name and eval_stat.name == "EulerFlow (Ours)":
                rotation = 0
                y_offset = 0.06
            plt.text(
                y_position,
                value - 0.11 + y_offset,
                f"{value:.4f}",
                ha="center",
                va="baseline",
                rotation=rotation,
                color="black" if not eval_stat.is_supervised() else "white",
                # set the font size to 2
                fontsize=2,
            )

        # Add the category name to the plot
        if "VEHICLE" in category_name:
            # Plot the name of the method vertically above the bar, rotated 90 degrees
            for idx, eval_stat in enumerate(eval_stats):
                y_position = centered_barchart_offset(idx, len(eval_stats), bar_width, bar_gap)
                value = _get_value(eval_stat, category_name)
                plt.text(
                    y_position,
                    value + 0.06,
                    eval_stat.name,
                    ha="center",
                    va="baseline",
                    rotation=90,
                )

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if "VEHICLE" in category_name:
            plt.ylabel("Dynamic Normalized EPE")

        savefig(save_folder, f"per_metacatagory_bar_{_clean_name(category_name).replace(' ', '_')}")
