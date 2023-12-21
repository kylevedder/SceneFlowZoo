import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

linewidth = 0.5
minor_tick_color = (0.9, 0.9, 0.9)

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


def savefig(save_folder : Path, name : str, pad: float = 0):
    save_folder = Path(save_folder)
    for ext in ['pdf', 'png']:
        outfile = save_folder / f"{name}.{ext}"
        print("Saving", outfile)
        plt.savefig(outfile, bbox_inches='tight', pad_inches=pad)
    plt.clf()


def savetable(save_folder : Path, name : str, content: List[List[Any]]):
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
