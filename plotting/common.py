"""Shared plotting style and helpers.  Plot scripts read result files only; they never
import the simulation package."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Validated categorical palette (fixed order, never cycled) and inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Single-hue sequential ramp (blue) for ordered quantities: band index, depth maps
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CMAP_BLUE = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd"] + BLUE_RAMP)


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 8.5,
    })


def ramp_color(i, n):
    """Color i of n from the sequential ramp (skipping the lightest steps for contrast)."""
    steps = BLUE_RAMP[2:]
    if n <= 1:
        return steps[-2]
    return steps[round(i * (len(steps) - 1) / (n - 1))]


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
