import numpy as np
from pathlib import Path
from loader_utils import *
import matplotlib
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple, Dict, Any, Optional
import shutil
from pathlib import Path

# Get path to methods from command line
parser = argparse.ArgumentParser()
parser.add_argument('results_folder', type=Path)
args = parser.parse_args()

assert args.results_folder.exists(
), f"Results folder {args.results_folder} does not exist"

save_folder = args.results_folder / "plots"
save_folder.mkdir(exist_ok=True, parents=True)


def set_font(size):
    matplotlib.rcParams.update({# Use mathtext, not LaTeX
                            'text.usetex': False,
                            # Use the Computer modern font
                            'font.family': 'serif',
                            'font.serif': ['cmr10'],
                            'font.size' : size,
                            'mathtext.fontset': 'cm',
                            # Use ASCII minus
                            'axes.unicode_minus': False,
                            })


def color(count, total_elements, intensity=1.3):
    start = 0.2
    stop = 0.7
    cm_subsection = np.linspace(start, stop, total_elements)
    #color = [matplotlib.cm.gist_earth(x) for x in cm_subsection][count]
    color = [matplotlib.cm.gist_earth(x) for x in cm_subsection][count]
    # Scale the color by intensity while leaving the 4th channel (alpha) unchanged
    return [min(x * intensity, 1) for x in color[:3]] + [color[3]]


def color2d(count_x, count_y, total_elements_x, total_elements_y):

    # Select the actual color, then scale along the intensity axis
    start = 1.7
    stop = 1
    intensity_scale = np.linspace(start, stop, total_elements_y)
    intensity = intensity_scale[count_y]
    return color(count_x, total_elements_x, intensity)


linewidth = 0.5
minor_tick_color = (0.9, 0.9, 0.9)


def grid():
    plt.grid(linewidth=linewidth / 2)
    plt.grid(which='minor',
             color=minor_tick_color,
             linestyle='--',
             alpha=0.7,
             clip_on=True,
             linewidth=linewidth / 4,
             zorder=0)


def savefig(name, pad: float = 0):
    for ext in ['pdf', 'png']:
        outfile = save_folder / f"{name}.{ext}"
        print("Saving", outfile)
        plt.savefig(outfile, bbox_inches='tight', pad_inches=pad)
    plt.clf()


def load_results(validation_folder: Path):
    print("Loading results from", validation_folder)
    config_folder = validation_folder / "configs"
    print()
    assert config_folder.exists(
    ), f"Config folder {config_folder} does not exist"
    result_lst = []
    for architecture_folder in sorted(config_folder.glob("*/")):
        if "bak" in architecture_folder.name:
            continue
        for result_file in architecture_folder.glob("*.pkl"):
            result_lst.append(
                ResultInfo(
                    architecture_folder.name + "_" +
                    result_file.stem.split(".")[0], result_file))

    return sorted(result_lst, key=lambda x: x.pretty_name())


print("Loading results...")
results = load_results(args.results_folder)
print("Done loading results.")
print(results)


def process_metacategory_counts(result):
    full_error_count = result['per_class_bucketed_error_count']
    metacatagory_results = {}
    # How do we do on vehicles by speed?
    for metacatagory in METACATAGORIES:
        category_names = METACATAGORIES[metacatagory]
        category_idxes = [CATEGORY_NAME_TO_IDX[cat] for cat in category_names]

        metacatagory_counts = full_error_count[category_idxes]
        # Sum up other axes
        metacatagory_counts = np.sum(metacatagory_counts, axis=(1, 2))
        metacatagory_results[metacatagory] = {}
        for category_result, category_name in zip(metacatagory_counts,
                                                  category_names):
            metacatagory_results[metacatagory][category_name] = category_result

    return metacatagory_results


def process_category_speed_counts(result):
    category_speed_counts_raw = result['per_class_bucketed_error_count'].sum(
        axis=2)
    category_speed_counts_normalized = category_speed_counts_raw / category_speed_counts_raw.sum(
        axis=1, keepdims=True)
    return category_speed_counts_normalized, category_speed_counts_raw


def bar_offset(pos_idx, num_pos, position_offset=0.2):
    """
    Compute the X offset for a bar plot given the number of bars and the
    index of the bar.
    """
    return -(num_pos + 1) / 2 * position_offset + position_offset * (pos_idx +
                                                                     1)


BAR_WIDTH = 0.1

num_metacatagories = len(METACATAGORIES)


def merge_dict_list(dict_list):
    result_dict = {}
    for d in dict_list:
        for k, v in d.items():
            if k not in result_dict:
                result_dict[k] = []
            result_dict[k].append(v)
    return result_dict


def plot_mover_nonmover_vs_error_by_category(results: List[ResultInfo],
                                             metacatagory, vmax):
    for result_idx, result in enumerate(results):
        metacatagory_epe_by_speed = result.get_metacatagory_epe_by_speed(
            metacatagory)
        xs = np.arange(len(metacatagory_epe_by_speed)) + bar_offset(
            result_idx, len(results), BAR_WIDTH)
        plt.bar(xs,
                metacatagory_epe_by_speed,
                label=result.pretty_name(),
                width=BAR_WIDTH,
                color=color(result_idx, len(results)),
                zorder=3)
    if metacatagory == "BACKGROUND":
        plt.xticks([-1, 0, 1], ["", "All", ""])
        plt.xlabel(" ")
        plt.legend()
    else:
        speed_buckets = result.speed_bucket_categories()
        plt.xticks(np.arange(len(speed_buckets)), [
            f"{l:0.1f}-" + (f"{u:0.1f}" if u != np.inf else "$\infty$")
            for l, u in speed_buckets
        ],
                   rotation=0)
        plt.xlabel("Speed Bucket (m/s)")
    plt.ylabel("Average EPE (m)")
    plt.ylim(0, vmax)
    grid()


def plot_nonmover_epe_overall(results: List[ResultInfo], vmax):
    for result_idx, result in enumerate(results):
        nonmover_epe = result.get_nonmover_point_epe()
        xs = bar_offset(result_idx, len(results), BAR_WIDTH)
        plt.bar(xs,
                nonmover_epe,
                label=result.pretty_name(),
                width=BAR_WIDTH,
                color=color(result_idx, len(results)),
                zorder=3)
    plt.ylabel("Average EPE (m)")
    plt.xticks([], [])
    plt.ylim(0, vmax)
    plt.legend()
    grid()


def plot_mover_epe_overall(results: List[ResultInfo], vmax):
    for result_idx, result in enumerate(results):
        mover_epe = result.get_mover_point_epe()
        xs = bar_offset(result_idx, len(results), BAR_WIDTH)
        plt.bar(xs,
                mover_epe,
                label=result.pretty_name(),
                width=BAR_WIDTH,
                color=color(result_idx, len(results)),
                zorder=3)
    plt.ylabel("Average EPE (m)")
    plt.xticks([], [])
    plt.ylim(0, vmax)
    plt.legend()
    grid()


def plot_metacatagory_epe_counts(results: List[ResultInfo]):

    for meta_idx, metacatagory in enumerate(sorted(METACATAGORIES.keys())):
        for result_idx, result in enumerate(results):
            # Each error bucket is a single bar in the stacked bar plot.
            metacatagory_epe_counts = result.get_metacatagory_count_by_epe(
                metacatagory)
            x_pos = meta_idx + bar_offset(
                len(results) - result_idx - 1, len(results), BAR_WIDTH)
            y_height = metacatagory_epe_counts.sum(
            ) / metacatagory_epe_counts.sum()
            bottom = 0
            for epe_idx, (epe_count,
                          (bucket_lower, bucket_upper)) in enumerate(
                              zip(metacatagory_epe_counts,
                                  result.speed_bucket_categories())):
                y_height = epe_count / metacatagory_epe_counts.sum()
                epe_color = color2d(result_idx, epe_idx, len(results),
                                    len(metacatagory_epe_counts))

                label = None
                if epe_idx == len(
                        metacatagory_epe_counts) - 1 and meta_idx == 0:
                    label = result.pretty_name()
                rect = plt.barh([x_pos], [y_height],
                                label=label,
                                height=BAR_WIDTH,
                                color=epe_color,
                                left=bottom)
                bottom += y_height
                # Draw text in middle of bar
                plt.text(bottom - y_height / 2,
                         x_pos,
                         f"{y_height * 100:0.1f}%",
                         ha="center",
                         va="center",
                         color="white",
                         fontsize=4)

    # xlabels to be the metacatagories
    plt.yticks(np.arange(len(METACATAGORIES)),
               [METACATAGORY_TO_SHORTNAME[e] for e in METACATAGORIES.keys()],
               rotation=0)
    plt.xticks(np.linspace(0, 1, 5),
               [f"{e}%" for e in np.linspace(0, 100, 5).astype(int)])
    plt.xlabel("Percentage of Endpoints Within Error Threshold")
    legend = plt.legend(loc="lower left", fancybox=False)
    # set the boarder of the legend artist to be transparent
    # legend.get_frame().set_edgecolor('none')
    plt.tight_layout()
    # plt.legend()


def plot_metacatagory_epe_counts_v15(results: List[ResultInfo]):

    for meta_idx, metacatagory in enumerate(sorted(METACATAGORIES.keys())):
        for result_idx, result in enumerate(results):
            # Each error bucket is a single bar in the stacked bar plot.
            metacatagory_epe_counts = result.get_metacatagory_count_by_epe(
                metacatagory)
            x_pos = meta_idx + bar_offset(
                len(results) - result_idx - 1, len(results), BAR_WIDTH)
            y_height = metacatagory_epe_counts.sum(
            ) / metacatagory_epe_counts.sum()
            bottom = 0
            metacatagory_epe_counts_subset = metacatagory_epe_counts[:2]
            for epe_idx, epe_count in enumerate(
                    metacatagory_epe_counts_subset):
                y_height = epe_count / metacatagory_epe_counts.sum()
                y_sum = metacatagory_epe_counts[:epe_idx + 1].sum(
                ) / metacatagory_epe_counts.sum()
                epe_color = color2d(result_idx, epe_idx, len(results),
                                    len(metacatagory_epe_counts_subset))

                label = None
                if epe_idx == len(
                        metacatagory_epe_counts_subset) - 1 and meta_idx == 0:
                    label = result.pretty_name()
                rect = plt.barh([x_pos], [y_height],
                                label=label,
                                height=BAR_WIDTH,
                                color=epe_color,
                                left=bottom)
                bottom += y_height
                # Draw text in middle of bar
                # plt.text(bottom - y_height / 2,
                #          x_pos,
                #          f"{y_sum * 100:0.1f}%",
                #          ha="center",
                #          va="center",
                #          color="white",
                #          fontsize=4)

    # xlabels to be the metacatagories
    plt.yticks(np.arange(len(METACATAGORIES)),
               [METACATAGORY_TO_SHORTNAME[e] for e in METACATAGORIES.keys()],
               rotation=0)
    plt.xticks(np.linspace(0, 1, 5),
               [f"{e}%" for e in np.linspace(0, 100, 5).astype(int)])
    plt.xlabel("Endpoints Within Error Threshold")
    legend = plt.legend(loc="lower left", fancybox=False)
    # set the boarder of the legend artist to be transparent
    # legend.get_frame().set_edgecolor('none')
    plt.tight_layout()
    # plt.legend()


def plot_metacatagory_epe_counts_v2(results: List[ResultInfo]):

    for meta_idx, metacatagory in enumerate(sorted(METACATAGORIES.keys())):
        for result_idx, result in enumerate(results):
            # Each error bucket is a single bar in the stacked bar plot.
            metacatagory_epe_counts = result.get_metacatagory_count_by_epe(
                metacatagory)
            x_pos_center = meta_idx + bar_offset(
                len(results) - result_idx - 1, len(results), BAR_WIDTH)
            y_height = metacatagory_epe_counts.sum(
            ) / metacatagory_epe_counts.sum()
            for epe_idx, epe_count in enumerate(metacatagory_epe_counts[:2]):
                x_offset = bar_offset(epe_idx, 2, BAR_WIDTH / 2)
                y_height = metacatagory_epe_counts[:epe_idx + 1].sum(
                ) / metacatagory_epe_counts.sum()
                epe_color = color2d(result_idx, epe_idx, len(results), 2)

                label = None
                if meta_idx == 0:
                    label = result.pretty_name()
                    if epe_idx == 0:
                        label += " (Strict)"
                    else:
                        label += " (Relaxed)"

                # if meta_idx == 0 and epe_idx == 0:
                #     label = result.pretty_name()

                plt.barh([x_pos_center + x_offset], [y_height],
                         label=label,
                         height=BAR_WIDTH / 2,
                         color=epe_color)

    # xlabels to be the metacatagories
    plt.yticks(np.arange(len(METACATAGORIES)),
               [METACATAGORY_TO_SHORTNAME[e] for e in METACATAGORIES.keys()],
               rotation=0)
    plt.xticks(np.linspace(0, 1, 5),
               [f"{e}%" for e in np.linspace(0, 100, 5).astype(int)])
    plt.xlabel("Percentage of Endpoints Within Error Threshold")
    legend = plt.legend(loc="lower left", fancybox=False)
    # set the boarder of the legend artist to be transparent
    # legend.get_frame().set_edgecolor('none')
    plt.tight_layout()
    # plt.legend()


def plot_metacatagory_epe_counts_v3(results: List[ResultInfo],
                                    epe_bucket_idx: int):

    for meta_idx, metacatagory in enumerate(sorted(METACATAGORIES.keys())):
        for result_idx, result in enumerate(results):
            # Each error bucket is a single bar in the stacked bar plot.
            metacatagory_epe_counts = result.get_metacatagory_count_by_epe(
                metacatagory)
            x_pos = meta_idx + bar_offset(
                len(results) - result_idx - 1, len(results), BAR_WIDTH)
            y_height = metacatagory_epe_counts.sum(
            ) / metacatagory_epe_counts.sum()

            y_height = metacatagory_epe_counts[:epe_bucket_idx + 1].sum(
            ) / metacatagory_epe_counts.sum()
            epe_color = color(result_idx, len(results))

            label = None
            if meta_idx == 0:
                label = result.pretty_name()
            plt.barh([x_pos], [y_height],
                     label=label,
                     height=BAR_WIDTH,
                     color=epe_color)

    # xlabels to be the metacatagories
    plt.yticks(np.arange(len(METACATAGORIES)),
               [METACATAGORY_TO_SHORTNAME[e] for e in METACATAGORIES.keys()],
               rotation=0)
    plt.xticks(np.linspace(0, 1, 5),
               [f"{e}%" for e in np.linspace(0, 100, 5).astype(int)])
    plt.xlabel("Endpoints Within Error Threshold")
    legend = plt.legend(loc="lower left", fancybox=False)
    # set the boarder of the legend artist to be transparent
    # legend.get_frame().set_edgecolor('none')
    plt.tight_layout()
    # plt.legend()


def table_latency(results: List[ResultInfo]):
    table = []
    for result in results:
        table.append([
            result.pretty_name(),
            f"{result.get_latency():0.4f}",
        ])
    return table


def plot_validation_pointcloud_size():
    validation_data_counts_path = args.results_folder / "validation_pointcloud_point_count.pkl"
    assert validation_data_counts_path.exists(
    ), f"Could not find {validation_data_counts_path}"
    point_cloud_counts = load_pickle(validation_data_counts_path)
    point_cloud_counts = np.array(point_cloud_counts)
    point_cloud_counts = np.sort(point_cloud_counts)

    print(f"Lowest 20 point cloud counts: {point_cloud_counts[:20]}")

    mean = np.mean(point_cloud_counts)
    std = np.std(point_cloud_counts)
    print(f"Mean point cloud count: {mean}, std: {std}")
    point_cloud_counts = point_cloud_counts[point_cloud_counts < 30000]
    # Make histogram of point cloud counts
    plt.hist(point_cloud_counts, bins=100, zorder=3, color=color(0, 1))
    plt.xlabel("Number of points")
    plt.ylabel("Number of point clouds")
    plt.tight_layout()


################################################################################

set_font(8)

for metacatagory in METACATAGORIES:
    plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6 / 2)
    plot_mover_nonmover_vs_error_by_category(results, metacatagory, vmax=0.3)
    print("saving", f"speed_vs_error_{metacatagory}")
    savefig(f"speed_vs_error_{metacatagory}")
    plt.clf()

################################################################################

plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6)
plot_nonmover_epe_overall(results, 0.3)
savefig(f"nonmover_epe_overall")

plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6)
plot_mover_epe_overall(results, 0.3)
savefig(f"mover_epe_overall")

################################################################################

plt.gcf().set_size_inches(5.5, 2.5)
plot_metacatagory_epe_counts(results)
savefig(f"epe_counts")

################################################################################

plt.gcf().set_size_inches(5.5, 2.5)
plot_metacatagory_epe_counts_v15(results)
savefig(f"epe_counts_v15")

################################################################################

plt.gcf().set_size_inches(5.5, 2.5)
plot_metacatagory_epe_counts_v2(results)
savefig(f"epe_counts_v2")

################################################################################

plt.gcf().set_size_inches(5.5 / 2, 2.5)
plot_metacatagory_epe_counts_v3(results, 0)
savefig(f"epe_counts_v3_strict")

plt.gcf().set_size_inches(5.5 / 2, 2.5)
plot_metacatagory_epe_counts_v3(results, 1)
savefig(f"epe_counts_v3_loose")

################################################################################

print(table_latency(results))

################################################################################

plt.gcf().set_size_inches(5.5, 5.5 / 1.6 / 2)
plot_validation_pointcloud_size()
grid()
savefig(f"validation_pointcloud_size", pad=0.02)

# def plot_meta_catagory_category_counts():
#     fig, subplot_axes = plt.subplots(1, num_metacatagories)

#     for result_idx, result in enumerate(results):
#         metacategory_counts = process_metacategory_counts(result.results)

#         for metacatagory, ax in zip(metacategory_counts, subplot_axes):
#             category_counts = metacategory_counts[metacatagory]
#             categories, counts = zip(*category_counts.items())
#             ax.set_title(metacatagory)
#             ax.bar(
#                 np.arange(len(categories)),
#                 counts,
#                 label=result.name.replace("fastflow3d_", " "),
#             )
#             ax.set_xticks(range(len(categories)), categories, rotation=90)

#     return fig

# def plot_category_speed_counts():
#     fig, axes = plt.subplots(1, len(results))
#     for result_idx, (result, ax) in enumerate(zip(results, axes)):
#         speed_counts_norm, speed_counts_raw = process_category_speed_counts(
#             result.results)
#         ax.matshow(speed_counts_norm)
#         for (i, j), z in np.ndenumerate(speed_counts_raw):
#             ax.text(j,
#                     i,
#                     '{}'.format(z),
#                     ha='center',
#                     va='center',
#                     bbox=dict(boxstyle='round',
#                               facecolor='white',
#                               edgecolor='0.3'))
#         ax.set_yticks(range(len(CATEGORY_NAMES)), CATEGORY_NAMES.values())

#     return fig

# fig = plot_meta_catagory_category_counts()
# fig.set_size_inches(20, 10)
# plt.tight_layout()
# fig.savefig(args.results_folder / "category_counts.png")
# plt.clf()

# fig = plot_category_speed_counts()
# fig.set_size_inches(10, 30)
# plt.tight_layout()
# fig.savefig(args.results_folder / "category_speed_counts.png")
# plt.clf()
