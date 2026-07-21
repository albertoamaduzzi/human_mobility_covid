"""
setup_mpl.py — Shared matplotlib/seaborn configuration for the paper figures.

Self-contained copy of ``src/set_mpl.py`` (same API, renamed for this
publication package) so ``paper/`` never has to import anything from
``src/``.

Usage:
    import setup_mpl
    setup_mpl.setup()           # call once per session/notebook
    # then use the constants/helpers below in your plot code
"""

import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Global setup
# ---------------------------------------------------------------------------

def setup():
    """Apply the paper-wide matplotlib/seaborn defaults."""
    sns.set_style("whitegrid")
    plt.rcParams["axes.grid"]       = True
    plt.rcParams["xtick.labelsize"] = FONTSIZE_TICK
    plt.rcParams["ytick.labelsize"] = FONTSIZE_TICK
    pd.set_option("display.max_rows", 100)
    warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# DPI / figure
# ---------------------------------------------------------------------------

DPI = 200


# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------

FONTSIZE_TITLE  = 25   # ax.set_title(...)
FONTSIZE_LABEL  = 24   # ax.set_xlabel / set_ylabel
FONTSIZE_TICK   = 22   # set_xticklabels / set_yticklabels
FONTSIZE_LEGEND = 18   # ax.legend(fontsize=...)
FONTSIZE_CBAR   = 25   # colorbar tick / label


# ---------------------------------------------------------------------------
# Line / marker parameters
# ---------------------------------------------------------------------------

LINEWIDTH       = 3      # theory curves
LINEWIDTH_THIN  = 1.5    # secondary / thin curves

SCATTER_SIZE_LG = 250    # large scatter (single-panel figures)
SCATTER_SIZE_MD = 150    # medium scatter (multi-panel figures)
SCATTER_SIZE_SM = 50     # small scatter (crowded panels)

ALPHA_THEORY    = 0.7    # line opacity — theory / mean-field data
ALPHA_SIM       = 0.3    # scatter opacity — stochastic simulations


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

def palette_teal(n=3):
    """Light teal palette with *n* shades (n <= 3, darkest last)."""
    pp = sns.light_palette("teal", n_colors=6)
    return [pp[3], pp[5], pp[5]][:n]


def palette_teal_slate(n=3):
    """Teal + darkslategrey blend used for radius = 3, 2, 1 curves."""
    pp  = sns.light_palette("teal", n_colors=6)
    ppp = sns.light_palette("darkslategrey", n_colors=6)
    return [pp[3], pp[5], ppp[5]][:n]


def palette_PuBuGn(n=3):
    """PuBuGn colormap sampled at high-value positions."""
    cmap = plt.cm.get_cmap("PuBuGn")
    return [cmap(s) for s in [0.65, 0.80, 0.95][:n]]


# Heatmap colormap
CMAP_HEATMAP = "viridis"


# ---------------------------------------------------------------------------
# Period colour palette -- muted, publication-safe replacement for
# matplotlib's default (bright) tab10 cycle, shared by every check_*/paper
# figure that overlays curves per period (pre/during/post restriction).
# ---------------------------------------------------------------------------

PERIOD_PALETTE_HEX = ["#4C78A8", "#F28E2B", "#59A14F"]   # blue, orange, green


def period_colors(periods) -> dict:
    """``{period: colour}`` -- one muted shade per period, sampled from a
    ``dark_palette`` built on that period's base hue (``PERIOD_PALETTE_HEX``),
    less saturated than matplotlib's default tab10 cycle."""
    cmaps = [sns.dark_palette(c, as_cmap=True) for c in PERIOD_PALETTE_HEX]
    return {p: cmaps[i % len(cmaps)](0.75) for i, p in enumerate(periods)}


def period_shades(periods) -> dict:
    """``{period: {"dark": colour, "light": colour}}`` -- a darker and a
    lighter shade of that period's base hue (``dark_palette`` /
    ``light_palette`` respectively), for plots that need two related series
    per period (e.g. sigma_x/sigma_y)."""
    out = {}
    for i, p in enumerate(periods):
        base = PERIOD_PALETTE_HEX[i % len(PERIOD_PALETTE_HEX)]
        out[p] = {
            "dark":  sns.dark_palette(base, as_cmap=True)(0.8),
            "light": sns.light_palette(base, as_cmap=True)(0.55),
        }
    return out


# ---------------------------------------------------------------------------
# Convenience: save figure
# ---------------------------------------------------------------------------

def save(fig, filename, fmt="png", transparent=True):
    """Save *fig* with paper-wide defaults (DPI, bbox, transparency).

    Parameters
    ----------
    fig        : matplotlib Figure
    filename   : output path (str or Path)
    fmt        : 'png' (default) or 'pdf'
    transparent: whether the background is transparent (default True)
    """
    fig.savefig(
        filename,
        format=fmt,
        dpi=DPI,
        transparent=transparent,
        bbox_inches="tight",
    )
