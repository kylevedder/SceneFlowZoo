import numpy as np
from pathlib import Path
from loader_utils import *
import matplotlib
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple, Dict, Any, Optional
import shutil

# Get path to methods from command line
parser = argparse.ArgumentParser()
parser.add_argument('results_folder', type=Path)
parser.add_argument('--dataset',
                    type=str,
                    default="argo",
                    choices=["argo", "waymo"])
args = parser.parse_args()

assert args.results_folder.exists(
), f"Results folder {args.results_folder} does not exist"

save_folder = args.results_folder / f"plots_{args.dataset}"
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


def color_map(rev: bool = False):
    # return 'gist_earth'
    if rev:
        return 'magma_r'
    return 'magma'


def color(count, total_elements, intensity=1.3):
    start = 0.2
    stop = 0.7

    colormap = matplotlib.cm.get_cmap(color_map())
    cm_subsection = np.linspace(start, stop, total_elements)
    #color = [matplotlib.cm.gist_earth(x) for x in cm_subsection][count]
    color = [colormap(x) for x in cm_subsection][count]
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


def grid(minor=True, axis='both'):
    plt.grid(linewidth=linewidth / 2, axis=axis)
    if minor:
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


def savetable(name, content: List[List[Any]]):
    outfile = save_folder / f"{name}.txt"

    def fmt(e):
        if type(e) == float or type(e) == np.float64 or type(
                e) == np.float32 or type(e) == np.float16:
            return f"{e:.3f}"
        return str(e)

    print("Saving", outfile)
    with open(outfile, 'w') as f:

        assert type(content) == list, "Table must be a list of rows"
        for row in content:
            assert type(row) == list, "Table rows must be lists"
            f.write(" & ".join([fmt(e) for e in row]) + "\\\\\n")


def load_results(validation_folder: Path,
                 dataset: str = args.dataset,
                 full_distance: str = "ALL"):
    print("Loading results from", validation_folder)
    config_folder = validation_folder / "configs"
    print()
    assert config_folder.exists(
    ), f"Config folder {config_folder} does not exist"
    result_lst = []
    for architecture_folder in sorted(config_folder.glob("*/")):
        if "bak" in architecture_folder.name:
            continue
        for result_file in (architecture_folder / dataset).glob("*.pkl"):
            result_lst.append(
                ResultInfo(architecture_folder.name + "_" +
                           ".".join(result_file.stem.split(".")[:-1]),
                           result_file,
                           full_distance=full_distance))

    return sorted(result_lst, key=lambda x: x.pretty_name())


print("Loading results...")
results = load_results(args.results_folder)
results_close = load_results(args.results_folder, full_distance="CLOSE")
results_far = load_results(args.results_folder, full_distance="FAR")
print("Done loading results.")
print(results)
assert len(
    results) > 0, f"No results found in {args.results_folder.absolute()}"


def plot_scaling():

    ours_results = [
        (1.0, 0.087),
        (0.5, 0.101),
        (0.2, 0.116),
        (0.1, 0.139),
        (0.01, 0.216),
    ]

    fastflow_results = [
        (1.0, 0.076),
        (0.5, 0.091),
        (0.2, 0.104),
        (0.1, 0.125),
    ]

    plt.plot(*zip(*ours_results), label="Ours", color='red', marker='o')
    plt.plot(*zip(*fastflow_results), label="FastFlow3D", color='black', marker='o')
    plt.xlabel("Dataset Fraction")
    plt.ylabel("Threeway EPE (m)")

    # Set x ticks
    plt.xticks([0.01, 0.1, 0.2, 0.5, 1.0], ["1%", "10%", "20%", "50%", "100%"])

    plt.legend()


def plot_speed_vs_performance_tradeoff(perf_error_bar: bool = True,
                                       runtime_error_bar: bool = True,
                                       gradient_bg: bool = True):
    runtimes = {
        'Ours': 29.33,
        'FastFlow3D': 29.33,
        'NSFP': 26285.0,
        'Chodosh': 26285.0 + 8996.4,
        'Gojcic': 6087.87,
        'Sim2Real': 99.3477,
        'EgoFlow': 2116.34,
        'PPWC': 79.4275,
        'FlowStep3D': 687.536,
        'ICP': 523.11,
    }

    runtimes_error_bar = {
        'Ours': 2.38,
        'FastFlow3D': 2.38,
        'NSFP': 18139.3,
        'Chodosh': 20247.7,
        'Gojcic': 1690.56,
        'Sim2Real': 13.88,
        'EgoFlow': 292.32,
        'PPWC': 2.20,
        'FlowStep3D': 3.13,
        'ICP': 169.34,
    }

    perf_error_bar_dims = {
        'Ours': (0.00, 0.002),
        'FastFlow3D': (0.002, 0.003),
    }

    points_processed = {
        'Ours': 52871.6,
        'FastFlow3D': 52871.6,
        'NSFP': 52871.6,
        'Chodosh': 52871.6,
        'Gojcic': 20000,
        'Sim2Real': 8192,
        'EgoFlow': 8192,
        'PPWC': 8192,
        'FlowStep3D': 8192,
        'ICP': 8192,
    }

    performance = {
        'Ours': 0.087,
        'NSFP': 0.068,
        'Chodosh': 0.061,
        'FastFlow3D': 0.076,
        'PPWC': 0.130,
        'FlowStep3D': 0.161,
        'Sim2Real': 0.157,
        'EgoFlow': 0.205,
        'Gojcic': 0.083,
        'ICP': 0.204,
    }

    uses_labels = {
        'Ours': False,
        'NSFP': False,
        'FastFlow3D': True,
        'PPWC': False,
        'FlowStep3D': False,
        'Sim2Real': False,
        'EgoFlow': False,
        'Gojcic': True,
        'Chodosh': False,
        'ICP': False,
    }

    horiz_offset = 2.5
    vert_offset = 6.0

    label_offset = {
        'ICP': (-horiz_offset, -vert_offset),
        'Sim2Real': (-horiz_offset, -vert_offset),
        'NSFP': (horiz_offset, vert_offset),
        'Gojcic': (-horiz_offset, vert_offset),
        'Chodosh': (horiz_offset, -vert_offset),
    }

    keys = runtimes.keys()
    runtimes = [runtimes[k] for k in keys]
    performance = [performance[k] for k in keys]
    shapes = ['x' if uses_labels[k] else 'o' for k in keys]
    alphas = [1.0 if uses_labels[k] else 1.0 for k in keys]
    gliph_colors = [
        'red' if k == 'Ours' else ('black' if uses_labels[k] else 'black')
        for k in keys
    ]
    text_colors = ['red' if k == 'Ours' else 'black' for k in keys]
    points = [points_processed[k] for k in keys]

    worst_runtime = max(runtimes)
    best_runtime = min(runtimes)
    worst_perf = max(performance)
    best_perf = min(performance)
    print(worst_runtime, best_runtime, worst_perf, best_perf)
    # breakpoint()

    plt.gca().axvspan(0, 100, alpha=0.08, facecolor='blue')

    def gradient_image(ax, direction=0.3, cmap_range=(0, 1), **kwargs):
        """
        Draw a gradient image based on a colormap.

        Parameters
        ----------
        ax : Axes
            The axes to draw on.
        direction : float
            The direction of the gradient. This is a number in
            range 0 (=vertical) to 1 (=horizontal).
        cmap_range : float, float
            The fraction (cmin, cmax) of the colormap that should be
            used for the gradient, where the complete colormap is (0, 1).
        **kwargs
            Other parameters are passed on to `.Axes.imshow()`.
            In particular, *cmap*, *extent*, and *transform* may be useful.
        """
        phi = direction * np.pi / 2
        v = np.array([np.cos(phi), np.sin(phi)])
        X = np.array([[v @ [1, 0], v @ [1, 1]], [v @ [0, 0], v @ [0, 1]]])
        a, b = cmap_range
        X = a + (b - a) / X.max() * X
        im = ax.imshow(X,
                       interpolation='bicubic',
                       clim=(0, 1),
                       aspect='auto',
                       **kwargs)
        return im

    if gradient_bg:
        gradient_image(plt.gca(),
                       direction=0.65,
                       extent=(0.0, 1, 0, 1),
                       transform=plt.gca().transAxes,
                       cmap=plt.get_cmap('Greys'),
                       alpha=0.35)

    plt.ylim(bottom=best_perf - 0.014, top=worst_perf + 0.008)
    plt.xlim(left=18, right=worst_runtime * 1.7)

    plt.text(43,
             worst_perf,
             'Real Time at 10Hz',
             color='blue',
             ha='center',
             rotation_mode='anchor')

    for key, runtime, perf, shape, alpha, point_count, gliph_color, text_color in zip(
            keys, runtimes, performance, shapes, alphas, points, gliph_colors,
            text_colors):

        fontweight = 'bold' if key == 'Ours' else 'normal'
        dot_scale = 35
        alignment = 'left' if runtime < 20000 else 'right'
        x_offset_sign = 1 if alignment == 'left' else -1
        plt.scatter(runtime,
                    perf,
                    marker=shape,
                    label=key,
                    color=gliph_color,
                    zorder=10,
                    s=dot_scale,
                    alpha=alpha,
                    edgecolors='none')
        if perf_error_bar:
            if key in perf_error_bar_dims:
                plt.errorbar(runtime,
                             perf,
                             yerr=np.array([perf_error_bar_dims[key]]).T,
                             color=gliph_color,
                             capsize=1.5,
                             zorder=0,
                             alpha=0.2)
        if runtime_error_bar:
            plt.errorbar(runtime,
                         perf,
                         xerr=runtimes_error_bar[key],
                         color=gliph_color,
                         capsize=1.5,
                         zorder=0,
                         alpha=0.2)
        # Annotate with name
        plt.annotate(f"{key} ({point_count / 1000.0:0.1f}k points)",
                     (runtime, perf),
                     xytext=(6 * x_offset_sign + label_offset.get(key,
                                                                  (0, 0))[0],
                             -2.75 + label_offset.get(key, (0, 0))[1]),
                     textcoords='offset points',
                     color=text_color,
                     ha=alignment,
                     fontweight=fontweight)
    # Set x axis log scale
    plt.xscale('log')
    # Make vertical bar at 100 on x axis
    # plt.axvline(100, color='blue', linestyle='--', linewidth=2, zorder=0)
    plt.xlabel("Runtime (ms)")
    plt.ylabel("Threeway EPE (m)")

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Draw arrow pointing from top right to bottom left
    plt.annotate('',
                 xy=(worst_runtime, worst_perf),
                 xytext=(worst_runtime / 4, worst_perf - 0.04),
                 color='gray',
                 arrowprops=dict(arrowstyle="<-", color='gray'))
    # Draw text along arrow
    plt.text(worst_runtime / 2,
             worst_perf - 0.017,
             'Better',
             color='gray',
             ha='center',
             rotation=35,
             rotation_mode='anchor')

    # Set plt y axis min to 0.05

    # plt.annotate('Worse', xy=(worst_runtime, worst_perf), color='gray')

    # grid(minor=False, axis='x')


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


BAR_WIDTH = 0.07

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


def table_speed_vs_error(results: List[ResultInfo]):
    # for metacatagory in METACATAGORIES:
    rows = []

    for result in results:
        row = [result.pretty_name()]
        for metacatagory in METACATAGORIES:

            metacatagory_epe_by_speed = result.get_metacatagory_epe_by_speed(
                metacatagory)
            print(metacatagory, metacatagory_epe_by_speed)
            row.extend(metacatagory_epe_by_speed.tolist())
        rows.append(row)

    return rows


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
        mover_epe = result.get_mover_point_all_epe()
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


def table_mover_nonmover_epe_overall(results: List[ResultInfo]):
    rows = []
    for result in results:
        row = [result.pretty_name()]
        row.append(result.get_nonmover_point_epe())
        row.append(result.get_mover_point_all_epe())
        rows.append(row)
    return rows


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
                if epe_idx == 0 and meta_idx == 0:
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
        table_row = [
            result.pretty_name(),
            f"{result.get_latency():0.4f}",
        ]
        table.append(table_row)
    return table


def table_epe(results: List[ResultInfo]):
    table_rows = []
    for result in results:
        foreground_counts = result.get_foreground_counts()
        relaxed_percentage = foreground_counts[:2].sum(
        ) / foreground_counts.sum()
        strict_percentage = foreground_counts[0] / foreground_counts.sum()

        def fmt(val):
            return f"${val:0.3f}$"

        epe_entries = [
            result.get_mover_point_dynamic_epe(),
            result.get_mover_point_static_epe(),
            result.get_nonmover_point_epe()
        ]
        average_epe = np.average(epe_entries)

        entries = [
            result.pretty_name(),
            fmt(average_epe), *[fmt(e) for e in epe_entries],
            fmt(relaxed_percentage),
            fmt(strict_percentage)
        ]
        table_rows.append(entries)
    return table_rows


def plot_validation_pointcloud_size(dataset: str, max_x=160000):
    validation_data_counts_path = args.results_folder / f"{dataset}_validation_pointcloud_point_count.pkl"
    assert validation_data_counts_path.exists(
    ), f"Could not find {validation_data_counts_path}"
    point_cloud_counts = load_pickle(validation_data_counts_path)
    point_cloud_counts = np.array(point_cloud_counts)
    point_cloud_counts = np.sort(point_cloud_counts)

    print(f"Lowest 20 point cloud counts: {point_cloud_counts[:20]}")

    mean = np.mean(point_cloud_counts)
    std = np.std(point_cloud_counts)
    print("VVVVVVVVVVVVVVVVVVVV")
    print(f"Mean point cloud count {dataset}: {mean}, std: {std}")
    print("^^^^^^^^^^^^^^^^^^^^")
    if dataset == "argo":
        point_cloud_counts = point_cloud_counts[point_cloud_counts < 100000]
    elif dataset == "waymo":
        point_cloud_counts = point_cloud_counts[point_cloud_counts < 200000]
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # Make histogram of point cloud counts
    plt.hist(point_cloud_counts, bins=100, zorder=3, color=color(0, 1))
    plt.xlabel("Number of points")
    plt.ylabel("Number of point clouds")
    plt.xlim(left=0, right=max_x)
    plt.gca().xaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.gca().yaxis.set_major_formatter(
        matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()


def plot_val_endpoint_error_distribution(source: str,
                                         use_log_scale: bool = False,
                                         unrotated: bool = False,
                                         ticks: bool = False):

    base_path = Path(
        f"/efs/argoverse2/val_{source}_unsupervised_vs_supervised_flow/")
    glob_str = "*_error_distribution.npy" if not unrotated else "*_error_distribution_unrotated.npy"
    error_distribution_files = sorted(base_path.glob(glob_str))
    npy_file_arr = np.array([
        load_npy(error_distribution_file, verbose=False)
        for error_distribution_file in error_distribution_files
    ])
    distribution = np.sum(npy_file_arr, axis=0)

    distribution_x = distribution.sum(axis=0)
    distribution_y = distribution.sum(axis=1)

    average_x = np.average(np.arange(distribution_x.shape[0]),
                           weights=distribution_x)
    average_y = np.average(np.arange(distribution_y.shape[0]),
                           weights=distribution_y)

    if (not unrotated) and (not use_log_scale):
        print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
        print(
            f"{source} Average x: {average_x}, average y: {average_y} is log scale: {use_log_scale} is rotated {not unrotated}"
        )
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    plot_mat = np.flip(distribution, axis=1).T
    if use_log_scale:
        plot_mat = np.log(plot_mat)
        print("++++++++++++++++++++++++++++")

        print("Num finite cells:", np.isfinite(plot_mat).sum())
        print(plot_mat[np.isfinite(plot_mat)].max(),
              plot_mat[np.isfinite(plot_mat)].min())
        plot_mat[~np.isfinite(plot_mat)] = -2
        print("++++++++++++++++++++++++++++")
    plt.imshow(plot_mat, cmap=color_map())
    grid_radius_meters = 1.5
    cells_per_meter = 50
    plt.xticks(
        np.linspace(0, grid_radius_meters * 2 * cells_per_meter,
                    int(grid_radius_meters * 2 + 1)),
        [
            f"{e}m" if ticks else ""
            for e in np.linspace(-grid_radius_meters, grid_radius_meters,
                                 int(grid_radius_meters * 2 + 1))
        ])
    plt.yticks(
        np.linspace(0, grid_radius_meters * 2 * cells_per_meter,
                    int(grid_radius_meters * 2 + 1)),
        [
            f"{-e}m" if ticks else ""
            for e in np.linspace(-grid_radius_meters, grid_radius_meters,
                                 int(grid_radius_meters * 2 + 1))
        ])


################################################################################

set_font(8)

vmax = 0.4

for metacatagory in METACATAGORIES:
    plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6 / 2)
    plot_mover_nonmover_vs_error_by_category(results, metacatagory, vmax=vmax)
    print("saving", f"speed_vs_error_{metacatagory}")
    savefig(f"speed_vs_error_{metacatagory}")
    plt.clf()

for metacatagory in METACATAGORIES:
    plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6 / 2)
    plot_mover_nonmover_vs_error_by_category(results_close,
                                             metacatagory,
                                             vmax=vmax)
    print("saving", f"speed_vs_error_{metacatagory}")
    savefig(f"speed_vs_error_{metacatagory}_close")
    plt.clf()

for metacatagory in METACATAGORIES:
    plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6 / 2)
    plot_mover_nonmover_vs_error_by_category(results_far,
                                             metacatagory,
                                             vmax=vmax)
    print("saving", f"speed_vs_error_{metacatagory}")
    savefig(f"speed_vs_error_{metacatagory}_far")
    plt.clf()

################################################################################

savetable("epe_table", table_epe(results_close))

################################################################################

plt.gcf().set_size_inches(6.5, 6.5 / 2.2)
plot_speed_vs_performance_tradeoff(gradient_bg=False)
savefig(f"speed_vs_performance_tradeoff")

plt.gcf().set_size_inches(6.5, 6.5 / 2.2)
plot_speed_vs_performance_tradeoff(gradient_bg=True)
savefig(f"speed_vs_performance_tradeoff_gradient")

################################################################################

plt.gcf().set_size_inches(6.5, 6.5 / 2.2)
plot_scaling()
savefig(f"scaling")

################################################################################

savetable("speed_vs_error", table_speed_vs_error(results))

################################################################################

error_dist_scalar = 0.6

# nsfp

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('nsfp', use_log_scale=True)
savefig(f"val_endpoint_error_distribution_log_nsfp")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('nsfp', use_log_scale=False)
savefig(f"val_endpoint_error_distribution_absolute_nsfp")

# Odom

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('odom', use_log_scale=True)
savefig(f"val_endpoint_error_distribution_log_odom")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('odom', use_log_scale=False)
savefig(f"val_endpoint_error_distribution_absolute_odom")

# Nearest neighbor

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('nearest_neighbor', use_log_scale=True)
savefig(f"val_endpoint_error_distribution_log_nearest_neighbor")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('nearest_neighbor', use_log_scale=False)
savefig(f"val_endpoint_error_distribution_absolute_nearest_neighbor")

# Chodosh

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('chodosh', use_log_scale=True)
savefig(f"val_endpoint_error_distribution_log_chodosh")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('chodosh', use_log_scale=False)
savefig(f"val_endpoint_error_distribution_absolute_chodosh")

# Supervised

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('supervised', use_log_scale=True)
savefig(f"val_endpoint_error_distribution_log_supervised")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('supervised', use_log_scale=False)
savefig(f"val_endpoint_error_distribution_absolute_supervised")

# Distilation

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('distilation', use_log_scale=True)
savefig(f"val_endpoint_error_distribution_log_distilation")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('distilation', use_log_scale=False)
savefig(f"val_endpoint_error_distribution_absolute_distilation")

# Chodosh distilation

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('chodosh_distilation', use_log_scale=True)
savefig(f"val_endpoint_error_distribution_log_chodosh_distilation")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('chodosh_distilation',
                                     use_log_scale=False)
savefig(f"val_endpoint_error_distribution_absolute_chodosh_distilation")

##### Unrotated

# nsfp

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('nsfp',
                                     use_log_scale=True,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_log_nsfp_unrotated")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('nsfp',
                                     use_log_scale=False,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_absolute_nsfp_unrotated")

# Odom

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('odom',
                                     use_log_scale=True,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_log_odom_unrotated")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('odom',
                                     use_log_scale=False,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_absolute_odom_unrotated")

# Nearest neighbor

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('nearest_neighbor',
                                     use_log_scale=True,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_log_nearest_neighbor_unrotated")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('nearest_neighbor',
                                     use_log_scale=False,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_absolute_nearest_neighbor_unrotated")

# Chodosh

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('chodosh',
                                     use_log_scale=True,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_log_chodosh_unrotated")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('chodosh',
                                     use_log_scale=False,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_absolute_chodosh_unrotated")

# Supervised

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('supervised',
                                     use_log_scale=True,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_log_supervised_unrotated")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('supervised',
                                     use_log_scale=False,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_absolute_supervised_unrotated")

# Distilation

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('distilation',
                                     use_log_scale=True,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_log_distilation_unrotated")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('distilation',
                                     use_log_scale=False,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_absolute_distilation_unrotated")

# Chodosh distilation

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('chodosh_distilation',
                                     use_log_scale=True,
                                     unrotated=True)
savefig(f"val_endpoint_error_distribution_log_chodosh_distilation_unrotated")

plt.gcf().set_size_inches(2.5 * error_dist_scalar, 2.5 * error_dist_scalar)
plot_val_endpoint_error_distribution('chodosh_distilation',
                                     use_log_scale=False,
                                     unrotated=True)
savefig(
    f"val_endpoint_error_distribution_absolute_chodosh_distilation_unrotated")

################################################################################

plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6)
plot_nonmover_epe_overall(results, 0.3)
savefig(f"nonmover_epe_overall")

plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6)
plot_mover_epe_overall(results, 0.3)
savefig(f"mover_epe_overall")

################################################################################

savetable("mover_nonmover_epe_overall",
          table_mover_nonmover_epe_overall(results))

################################################################################

plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6)
plot_nonmover_epe_overall(results_close, 0.3)
savefig(f"nonmover_epe_overall_close")

plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6)
plot_mover_epe_overall(results_close, 0.3)
savefig(f"mover_epe_overall_close")

plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6)
plot_nonmover_epe_overall(results_far, 0.4)
savefig(f"nonmover_epe_overall_far")

plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6)
plot_mover_epe_overall(results_far, 0.4)
savefig(f"mover_epe_overall_far")

################################################################################

plt.gcf().set_size_inches(5.5, 2.5)
plot_metacatagory_epe_counts(results)
savefig(f"epe_counts")

plt.gcf().set_size_inches(5.5, 2.5)
plot_metacatagory_epe_counts(results_close)
savefig(f"epe_counts_close")

plt.gcf().set_size_inches(5.5, 2.5)
plot_metacatagory_epe_counts(results_far)
savefig(f"epe_counts_close_far")

################################################################################

plt.gcf().set_size_inches(5.5, 2.5)
plot_metacatagory_epe_counts_v15(results)
savefig(f"epe_counts_v15")

plt.gcf().set_size_inches(5.5, 2.5)
plot_metacatagory_epe_counts_v15(results_close)
savefig(f"epe_counts_v15_close")

plt.gcf().set_size_inches(5.5, 2.5)
plot_metacatagory_epe_counts_v15(results_far)
savefig(f"epe_counts_v15_far")

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

plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6 / 2)
plot_validation_pointcloud_size("argo", max_x=90000)
grid()
savefig(f"validation_pointcloud_size_argo", pad=0.02)

plt.gcf().set_size_inches(5.5 / 2, 5.5 / 1.6 / 2)
plot_validation_pointcloud_size("waymo")
grid()
savefig(f"validation_pointcloud_size_waymo", pad=0.02)

################################################################################

savetable("latency_table", table_latency(results))
