"""
plot_change_mobility_covid.py
==============================
Every figure in the paper (figure_1 .. figure_13), reproduced from the
flat ``figure_<i>.csv`` files in ``paper/data/`` instead of the raw
consolidated parquet trees the ``check_*``/``distance_check`` scripts in
``src/`` read from. Fully self-contained: the only imports are
``numpy``/``pandas``/``matplotlib``/``scipy`` plus this package's own
``setup_mpl.py`` and ``metrics_change_mobility_covid.py`` — no ``src/``,
no S3, no ``ParquetStore``.

Each plot below is a byte-for-byte visual match of the corresponding
figure produced in ``src/check_*``/``src/distance_check`` (same colours,
fonts, layout, legends) — only the data-loading step changed, from
"scan a consolidated parquet tree and aggregate" to "read one flat CSV".
See each function's docstring for which ``check_*`` script it mirrors and
the exact CSV column contract it expects (also documented in
``paper/README.md``).

Usage
-----
    import plot_change_mobility_covid as pcmc
    pcmc.plot_figure_1("data/figure_1.csv")     # -> figures/figure_1.png

    # or regenerate every figure from the CLI:
    python plot_change_mobility_covid.py
    python plot_change_mobility_covid.py --only figure_1 figure_8 --data-dir data --out-dir figures
"""

from __future__ import annotations

import argparse
from datetime import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import setup_mpl
from metrics_change_mobility_covid import power_law_fit

DIR_HERE    = Path(__file__).resolve().parent
DIR_DATA    = DIR_HERE / "data"
DIR_FIGURES = DIR_HERE / "figures"

# ---------------------------------------------------------------------------
# Shared constants (paper convention -- see src/constants.py for the
# production copies these mirror)
# ---------------------------------------------------------------------------

PERIOD_NAMES = ["15 jan - 15 march", "15 march - 15 may", "15 may - sept"]
PERIOD_DIVISION = [
    _dt(2020, 1, 15), _dt(2020, 3, 15), _dt(2020, 5, 15), _dt(2020, 9, 30),
]
RURALITY_LEVELS = ["rural", "urban"]
PARTY_NAMES     = ["Democratic", "Republican"]
RURAL_COLOR     = {"rural": "black", "urban": "grey"}
PARTY_COLOR     = {"Democratic": "black", "Republican": "grey"}

ETYPES = ["random_entropy", "uncorrelated_entropy", "real_entropy", "scaled_entropy"]
ETYPE_XLABEL = {
    "random_entropy":       "Random entropy",
    "uncorrelated_entropy": "Uncorrelated entropy",
    "real_entropy":         "Real entropy",
    "scaled_entropy":       r"Real / Random entropy",
}
SHARED_ENTROPY_XLIM = (0.0, 9.0)
SHARED_ENTROPY_YLIM = (0.0, 0.8)

# Approximate exploration exponents (mu) per period as a fallback estimate
# if the OLS power-law fit fails -- same convention/values as the
# production plotter.plot_St's _MU_FALLBACK.
_MU_FALLBACK_VALUES = [0.668, 0.523, 0.668]


def _present_periods(df: pd.DataFrame, col: str = "period") -> list:
    present = set(df[col])
    return [p for p in PERIOD_NAMES if p in present] or sorted(present)


def _cycle_colors(periods: list) -> dict:
    """Matplotlib's own (default tab10) axes.prop_cycle colours, one per
    period -- used by rg_krg / st_and_frequency / entropy, matching the
    corresponding check_* script's own (non-muted) colour choice."""
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {p: cycle[i % len(cycle)] for i, p in enumerate(periods)}


def _save(fig, out_dir: Path, name: str, fmt: str = "png") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.{fmt}"
    setup_mpl.save(fig, out_path, fmt=fmt)
    plt.close(fig)
    return out_path


def _grid_from_sparse(
    sub: pd.DataFrame, xcol: str, ycol: str, valcol: str,
    x_centers: np.ndarray, y_centers: np.ndarray,
) -> np.ndarray:
    """
    Rebuild a dense ``(len(y_centers), len(x_centers))`` grid from a
    sparse long-format table (zero/omitted cells default to 0) --
    ``extract_figure_data.py`` drops zero-density/zero-count cells to
    keep figure_2/4/9/11.csv small, so this is the inverse operation.
    Bin membership is resolved by nearest-centre (robust to the tiny
    float round-trip noise introduced by writing/reading a CSV), not
    exact float equality.
    """
    nx, ny = len(x_centers), len(y_centers)
    H = np.zeros((ny, nx))
    if sub.empty:
        return H
    x_width = (x_centers[-1] - x_centers[0]) / (nx - 1) if nx > 1 else 1.0
    y_width = (y_centers[-1] - y_centers[0]) / (ny - 1) if ny > 1 else 1.0
    xi = np.clip(np.round((sub[xcol].to_numpy() - x_centers[0]) / x_width).astype(int), 0, nx - 1)
    yi = np.clip(np.round((sub[ycol].to_numpy() - y_centers[0]) / y_width).astype(int), 0, ny - 1)
    np.add.at(H, (yi, xi), sub[valcol].to_numpy())
    return H


# ===========================================================================
# figure_1 / figure_8 -- rg_and_dist
#   Mirrors src/distance_check/visualiza_rg_and_dist_distribution.py
#   CSV columns: panel [radius_of_gyration|jump_length], period, bin_center,
#                pdf, fit_alpha, fit_sigma, fit_xmin, fit_fraction
# ===========================================================================

def _draw_rg_panel(ax, df_rg: pd.DataFrame, colors: dict) -> None:
    for period in colors:
        sub = df_rg[df_rg["period"] == period].sort_values("bin_center")
        if sub.empty:
            continue
        ax.loglog(sub["bin_center"], sub["pdf"], linewidth=setup_mpl.LINEWIDTH, color=colors[period])
    ax.set_xlabel(r"$R_g$ (km)", fontsize=setup_mpl.FONTSIZE_LABEL)
    ax.set_ylabel("PDF", fontsize=setup_mpl.FONTSIZE_LABEL)


def _draw_distance_panel(ax, df_dist: pd.DataFrame, colors: dict, exponent_only: bool = True) -> None:
    for period in colors:
        sub = df_dist[df_dist["period"] == period].sort_values("bin_center")
        if sub.empty:
            continue
        color = colors[period]
        alpha_v  = float(sub["fit_alpha"].iloc[0])
        sigma_v  = float(sub["fit_sigma"].iloc[0])
        xmin     = float(sub["fit_xmin"].iloc[0])
        fraction = float(sub["fit_fraction"].iloc[0])
        ax.scatter(sub["bin_center"], sub["pdf"], s=16, color=color, alpha=0.35,
                   edgecolors="none", zorder=2, label=None)
        if np.isfinite(alpha_v) and np.isfinite(xmin) and sub["bin_center"].max() > xmin:
            x_fit = np.logspace(np.log10(xmin), np.log10(sub["bin_center"].max()), 200)
            C = fraction * (alpha_v - 1) * xmin ** (alpha_v - 1)
            lbl = (rf"$\alpha$ = {alpha_v:.3f} $\pm$ {sigma_v:.3f}" if exponent_only else
                   rf"{period}: $\alpha$ = {alpha_v:.3f} $\pm$ {sigma_v:.3f}, $x_{{min}}$ = {xmin:.2f} km")
            ax.loglog(x_fit, C * x_fit ** (-alpha_v), color=color, linestyle=(0, (6, 3)),
                      linewidth=setup_mpl.LINEWIDTH + 1.5, zorder=5, dash_capstyle="round", label=lbl)
        else:
            lbl = "fit failed" if exponent_only else f"{period}: fit failed"
            ax.plot([], [], color=color, linewidth=setup_mpl.LINEWIDTH, label=lbl)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Jump length (km)", fontsize=setup_mpl.FONTSIZE_LABEL)
    ax.set_ylabel("PDF", fontsize=setup_mpl.FONTSIZE_LABEL)
    ax.legend(fontsize=setup_mpl.FONTSIZE_LEGEND)


def draw_rg_and_dist(df: pd.DataFrame):
    periods = _present_periods(df)
    colors  = setup_mpl.period_colors(periods)
    fig, (ax_rg, ax_dist) = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    _draw_rg_panel(ax_rg, df[df["panel"] == "radius_of_gyration"], colors)
    _draw_distance_panel(ax_dist, df[df["panel"] == "jump_length"], colors, exponent_only=True)
    fig.tight_layout()
    return fig


def plot_figure_1(csv_path=DIR_DATA / "figure_1.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_1 -- rg_and_dist, California."""
    return _save(draw_rg_and_dist(pd.read_csv(csv_path)), out_dir, "figure_1", fmt)


def plot_figure_8(csv_path=DIR_DATA / "figure_8.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_8 -- rg_and_dist, Massachusetts."""
    return _save(draw_rg_and_dist(pd.read_csv(csv_path)), out_dir, "figure_8", fmt)


# ===========================================================================
# figure_2 / figure_9 -- rg_krg
#   Mirrors src/check_rg_krg/plot_rg_krg.py
#   CSV columns: period, k, rg_km, krg_km, density (zero-density cells omitted)
# ===========================================================================

RG_KRG_RANGE = (0.1, 1500.0)   # km -- same as check_rg_krg.plot_rg_krg.RG_RANGE
RG_KRG_NBINS = 40


def draw_rg_krg(df: pd.DataFrame):
    edges   = np.linspace(RG_KRG_RANGE[0], RG_KRG_RANGE[1], RG_KRG_NBINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    Xm, Ym  = np.meshgrid(centers, centers)

    periods = _present_periods(df)
    list_k  = sorted(int(k) for k in df["k"].unique())
    cmaps   = {p: mcolors.LinearSegmentedColormap.from_list(f"{p}_cmap", ["white", c])
               for p, c in _cycle_colors(periods).items()}

    n_rows, n_cols = len(periods), len(list_k)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows + 1),
                             sharex=True, sharey=True, squeeze=False, constrained_layout=True)

    for row_idx, p in enumerate(periods):
        row_H = {}
        for k in list_k:
            sub = df[(df["period"] == p) & (df["k"] == k)]
            row_H[k] = _grid_from_sparse(sub, "rg_km", "krg_km", "density", centers, centers)
        pos_vals = [H[H > 0] for H in row_H.values()]
        pos_vals = [v for v in pos_vals if v.size]
        if not pos_vals:
            for col_idx in range(n_cols):
                axes[row_idx][col_idx].set_visible(False)
            continue
        norm = mcolors.LogNorm(vmin=min(v.min() for v in pos_vals), vmax=max(v.max() for v in pos_vals))

        pcm_row = None
        for col_idx, k in enumerate(list_k):
            ax = axes[row_idx][col_idx]
            H = row_H[k]
            if H.max() <= 0:
                ax.set_visible(False)
                continue
            pcm_row = ax.pcolormesh(Xm, Ym, H, norm=norm, cmap=cmaps[p], shading="auto")
            if row_idx == 0:
                ax.set_title(rf"$k={k}$", fontsize=setup_mpl.FONTSIZE_TITLE)
            if col_idx == 0:
                ax.set_ylabel(r"$k\text{-}R_g$ (km)", fontsize=setup_mpl.FONTSIZE_LABEL)
            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$R_g$ (km)", fontsize=setup_mpl.FONTSIZE_LABEL)

        if pcm_row is not None:
            cbar = fig.colorbar(pcm_row, ax=axes[row_idx].tolist(), location="right",
                                fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=setup_mpl.FONTSIZE_TICK)

    return fig


def plot_figure_2(csv_path=DIR_DATA / "figure_2.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_2 -- rg_krg, California."""
    return _save(draw_rg_krg(pd.read_csv(csv_path)), out_dir, "figure_2", fmt)


def plot_figure_9(csv_path=DIR_DATA / "figure_9.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_9 -- rg_krg, Massachusetts."""
    return _save(draw_rg_krg(pd.read_csv(csv_path)), out_dir, "figure_9", fmt)


# ===========================================================================
# figure_3 / figure_10 -- st_and_frequency
#   Mirrors src/check_frequency_st/plot_frequency_st.py
#   CSV columns: panel [st|frequency], period, time_step, mean_st, std_st,
#                rank, mean_frequency
# ===========================================================================

DEFAULT_MAX_RANK = 9   # same as check_frequency_st.plot_frequency_st.DEFAULT_MAX_RANK


def _draw_st_panel(ax, df_st: pd.DataFrame, colors: dict) -> None:
    periods = list(colors)
    mu_fallback = {p: _MU_FALLBACK_VALUES[i % len(_MU_FALLBACK_VALUES)] for i, p in enumerate(periods)}
    for p in periods:
        sub = df_st[df_st["period"] == p].sort_values("time_step")
        if sub.empty:
            continue
        l_mean = sub["mean_st"].to_numpy()
        t_arr  = np.arange(len(l_mean))
        mask   = np.isfinite(l_mean)
        l_mean, t_arr = l_mean[mask], t_arr[mask]

        color = colors[p]
        ax.scatter(t_arr, l_mean, s=20, alpha=setup_mpl.ALPHA_SIM, color=color)
        try:
            slope, std_err, _r, _i = power_law_fit(t_arr, l_mean)
            u_spacing = t_arr.astype(float) ** slope
            ax.loglog(t_arr[10:], u_spacing[10:], linestyle="dashed", color=color,
                      linewidth=setup_mpl.LINEWIDTH,
                      label=rf"$\mu$ = {slope:.3f} $\pm$ {std_err:.3f}")
        except Exception:
            mu_fb = mu_fallback.get(p, 0.6)
            u_spacing = t_arr.astype(float) ** mu_fb
            ax.loglog(t_arr[10:], u_spacing[10:], linestyle="dashed", color=color,
                      linewidth=setup_mpl.LINEWIDTH, label=rf"$\mu$ ~ {mu_fb:.3f} (fallback)")
    ax.set_xlabel("t (h)", fontsize=setup_mpl.FONTSIZE_LABEL)
    ax.set_ylabel("S(t)", fontsize=setup_mpl.FONTSIZE_LABEL)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=setup_mpl.FONTSIZE_LEGEND, loc="upper left")


def _draw_frequency_panel(ax, df_freq: pd.DataFrame, colors: dict, max_rank: int = DEFAULT_MAX_RANK) -> None:
    periods = list(colors)
    x = np.arange(max_rank)
    w, bar_w = 0.3, 0.3 * 0.85
    for idx, p in enumerate(periods):
        sub  = df_freq[df_freq["period"] == p].sort_values("rank")
        vals = sub["mean_frequency"].tolist()
        ax.bar(x[:len(vals)] + idx * w + (w - bar_w) / 2, vals, width=bar_w, color=colors[p])
    ax.set_xlabel("Rank", fontsize=setup_mpl.FONTSIZE_LABEL)
    ax.set_ylabel(r"$\langle k \rangle$", fontsize=setup_mpl.FONTSIZE_LABEL)


def draw_st_and_frequency(df: pd.DataFrame):
    periods = _present_periods(df)
    colors  = _cycle_colors(periods)
    fig, (ax_st, ax_freq) = plt.subplots(1, 2, figsize=(16, 6))
    _draw_st_panel(ax_st, df[df["panel"] == "st"], colors)
    _draw_frequency_panel(ax_freq, df[df["panel"] == "frequency"], colors)
    fig.tight_layout()
    return fig


def plot_figure_3(csv_path=DIR_DATA / "figure_3.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_3 -- st_and_frequency, California."""
    return _save(draw_st_and_frequency(pd.read_csv(csv_path)), out_dir, "figure_3", fmt)


def plot_figure_10(csv_path=DIR_DATA / "figure_10.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_10 -- st_and_frequency, Massachusetts."""
    return _save(draw_st_and_frequency(pd.read_csv(csv_path)), out_dir, "figure_10", fmt)


# ===========================================================================
# figure_4 / figure_11 -- gonzalez (2-D density + x/sigma_x~=0 profile)
#   Mirrors src/check_gonzalez/plot_gonzalez_old.py (plot_gonzalez_old)
#   CSV columns: period, x_norm, y_norm, count (zero-count cells omitted)
# ===========================================================================

GONZALEZ_XRANGE = (-1.5, 1.5)
GONZALEZ_YRANGE = (-2.2, 2.2)
GONZALEZ_NBINS  = 40


def draw_gonzalez(df: pd.DataFrame):
    xmin, xmax = GONZALEZ_XRANGE
    ymin, ymax = GONZALEZ_YRANGE
    nbins = GONZALEZ_NBINS
    x_edges   = np.linspace(xmin, xmax, nbins + 1)
    y_edges   = np.linspace(ymin, ymax, nbins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xm, Ym = np.meshgrid(x_centers, y_centers)

    periods = _present_periods(df)
    Hs = {p: _grid_from_sparse(df[df["period"] == p], "x_norm", "y_norm", "count", x_centers, y_centers)
          for p in periods}

    pos_vals = [H[H > 0] for H in Hs.values()]
    pos_vals = [v for v in pos_vals if v.size]
    norm_sh = mcolors.LogNorm(vmin=min(v.min() for v in pos_vals), vmax=max(v.max() for v in pos_vals))

    n_periods = len(periods)
    FS_LABEL = setup_mpl.FONTSIZE_LABEL + 6
    FS_TICK  = setup_mpl.FONTSIZE_TICK + 6
    FS_CBAR  = setup_mpl.FONTSIZE_CBAR + 6

    fig, axes = plt.subplots(2, n_periods, figsize=(8 * n_periods, 13),
                             sharex="row", sharey="row", squeeze=False, constrained_layout=True)
    colors = setup_mpl.period_colors(periods)

    pcm_ref = None
    for idx, p in enumerate(periods):
        ax_2d, ax_profile = axes[0][idx], axes[1][idx]
        H = Hs[p]
        if H.max() <= 0:
            ax_2d.set_visible(False)
            ax_profile.set_visible(False)
            continue

        pcm = ax_2d.pcolormesh(Xm, Ym, H, norm=norm_sh, cmap=plt.cm.jet, shading="auto")
        pcm_ref = pcm
        ax_2d.tick_params(labelsize=FS_TICK)
        if idx == 0:
            ax_2d.set_ylabel(r"$y / \sigma_y$", fontsize=FS_LABEL)

        profile = H[:, nbins // 2].astype(float)
        profile[profile == 0] = np.nan
        ax_profile.semilogy(y_centers, profile, linewidth=setup_mpl.LINEWIDTH + 1.5, color=colors[p])
        ax_profile.set_xlabel(r"$y / \sigma_y$", fontsize=FS_LABEL)
        ax_profile.tick_params(labelsize=FS_TICK)
        ax_profile.grid(True, alpha=0.3)
        if idx == 0:
            ax_profile.set_ylabel(
                r"$\Phi\!\left(\frac{y}{\sigma_y}\,\middle|\,\frac{x}{\sigma_x}=0\right)$", fontsize=FS_LABEL,
            )

    if pcm_ref is not None:
        cbar = fig.colorbar(pcm_ref, ax=axes.ravel().tolist(), orientation="horizontal",
                            location="top", fraction=0.03, pad=0.02)
        cbar.set_label("Count", fontsize=FS_CBAR)
        cbar.ax.tick_params(labelsize=FS_TICK)

    return fig


def plot_figure_4(csv_path=DIR_DATA / "figure_4.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_4 -- gonzalez, California."""
    return _save(draw_gonzalez(pd.read_csv(csv_path)), out_dir, "figure_4", fmt)


def plot_figure_11(csv_path=DIR_DATA / "figure_11.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_11 -- gonzalez, Massachusetts."""
    return _save(draw_gonzalez(pd.read_csv(csv_path)), out_dir, "figure_11", fmt)


# ===========================================================================
# figure_5 / figure_12 -- sigmaxy
#   Mirrors src/check_gonzalez/plot_gonzalez_old.py (plot_sigmaxy_old)
#   CSV columns: period, axis [sigma_x|sigma_y], bin_center, pdf, mean_km,
#                mean_pct_change_vs_baseline
# ===========================================================================

def draw_sigmaxy(df: pd.DataFrame):
    periods = _present_periods(df)
    baseline = periods[0]
    axis2mean_label = {"sigma_x": r"$\langle\sigma_x\rangle$", "sigma_y": r"$\langle\sigma_y\rangle$"}
    axis2key = {"sigma_x": "dark", "sigma_y": "light"}
    shades = setup_mpl.period_shades(periods)

    FS_LABEL  = setup_mpl.FONTSIZE_LABEL + 6
    FS_TICK   = setup_mpl.FONTSIZE_TICK + 6
    FS_LEGEND = setup_mpl.FONTSIZE_LEGEND + 4

    n_periods = len(periods)
    fig, axes = plt.subplots(1, n_periods, figsize=(8 * n_periods, 5.5), sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    for idx, p in enumerate(periods):
        ax = axes[0][idx]
        axis2color = {a: shades[p][axis2key[a]] for a in ["sigma_x", "sigma_y"]}
        annotations = []
        for axis in ["sigma_x", "sigma_y"]:
            sub = df[(df["period"] == p) & (df["axis"] == axis)].sort_values("bin_center")
            if not sub.empty:
                ax.loglog(sub["bin_center"], sub["pdf"], linewidth=setup_mpl.LINEWIDTH + 1, color=axis2color[axis])
            mean_v = float(sub["mean_km"].iloc[0]) if not sub.empty else float("nan")
            if not np.isfinite(mean_v):
                continue
            ax.axvline(mean_v, color=axis2color[axis], linestyle=":",
                       linewidth=setup_mpl.LINEWIDTH_THIN + 1.5, alpha=0.8)
            txt = rf"{axis2mean_label[axis]} = {mean_v:.2f} km"
            if p != baseline:
                pct = float(sub["mean_pct_change_vs_baseline"].iloc[0]) if not sub.empty else float("nan")
                if np.isfinite(pct):
                    txt += rf"  ({pct:+.0f}%)"
            annotations.append((axis, txt))
        for j, (axis, txt) in enumerate(annotations):
            ax.text(0.03, 0.05 + 0.09 * j, txt, transform=ax.transAxes, ha="left", va="bottom",
                    color="black", fontsize=FS_LEGEND)
        ax.set_xlabel(r"$\sigma$ (km)", fontsize=FS_LABEL)
        if idx == 0:
            ax.set_ylabel("PDF", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)

    return fig


def plot_figure_5(csv_path=DIR_DATA / "figure_5.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_5 -- sigmaxy, California."""
    return _save(draw_sigmaxy(pd.read_csv(csv_path)), out_dir, "figure_5", fmt)


def plot_figure_12(csv_path=DIR_DATA / "figure_12.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_12 -- sigmaxy, Massachusetts."""
    return _save(draw_sigmaxy(pd.read_csv(csv_path)), out_dir, "figure_12", fmt)


# ===========================================================================
# figure_6 / figure_13 -- entropy
#   Mirrors src/check_entropy/plot_entropy_new.py
#   CSV columns: period, entropy_type, bin_center, pdf
# ===========================================================================

def draw_entropy(df: pd.DataFrame):
    periods = _present_periods(df)
    colors  = _cycle_colors(periods)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)
    for idx, etype in enumerate(ETYPES):
        ax = axes[idx // 2][idx % 2]
        for p in periods:
            sub = df[(df["period"] == p) & (df["entropy_type"] == etype)].sort_values("bin_center")
            if sub.empty:
                continue
            ax.plot(sub["bin_center"], sub["pdf"], linewidth=setup_mpl.LINEWIDTH, color=colors[p])
        ax.set_xlabel(ETYPE_XLABEL[etype], fontsize=setup_mpl.FONTSIZE_LABEL)
        if idx % 2 == 0:
            ax.set_ylabel("PDF", fontsize=setup_mpl.FONTSIZE_LABEL)
        if etype != "scaled_entropy":
            ax.set_xlim(SHARED_ENTROPY_XLIM)
            ax.set_ylim(SHARED_ENTROPY_YLIM)
    plt.tight_layout()
    return fig


def plot_figure_6(csv_path=DIR_DATA / "figure_6.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_6 -- entropy, California."""
    return _save(draw_entropy(pd.read_csv(csv_path)), out_dir, "figure_6", fmt)


def plot_figure_13(csv_path=DIR_DATA / "figure_13.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_13 -- entropy, Massachusetts."""
    return _save(draw_entropy(pd.read_csv(csv_path)), out_dir, "figure_13", fmt)


# ===========================================================================
# figure_7 -- weekly_rg (California only)
#   Mirrors src/check_rg_weekly/plot_weekly_rg.py
#   CSV columns: panel [rurality|party], week, stratum, mean_rg, std_rg
# ===========================================================================

def _parse_week(w: str):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return _dt.strptime(w, fmt)
        except ValueError:
            pass
    return w


def _add_period_bands(ax) -> None:
    for i in range(len(PERIOD_NAMES)):
        ax.axvspan(PERIOD_DIVISION[i], PERIOD_DIVISION[i + 1],
                   color=setup_mpl.PERIOD_PALETTE_HEX[i % len(setup_mpl.PERIOD_PALETTE_HEX)],
                   alpha=0.12, zorder=0)


def _add_boundary_lines(ax) -> None:
    for boundary in PERIOD_DIVISION[1:-1]:
        ax.axvline(boundary, color="black", linestyle="--",
                   linewidth=setup_mpl.LINEWIDTH_THIN, alpha=0.7, zorder=1)


def draw_weekly_rg(df: pd.DataFrame):
    df_rur   = df[df["panel"] == "rurality"]
    df_party = df[df["panel"] == "party"]

    sorted_weeks_rur   = sorted(df_rur["week"].unique(), key=_parse_week) if not df_rur.empty else []
    sorted_weeks_party = sorted(df_party["week"].unique(), key=_parse_week) if not df_party.empty else []

    fig, (ax_rur, ax_party) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    _add_period_bands(ax_rur)
    _add_period_bands(ax_party)

    for rur in RURALITY_LEVELS:
        sub = df_rur[df_rur["stratum"] == rur].set_index("week").reindex(sorted_weeks_rur)
        week_dts = [_parse_week(w) for w in sorted_weeks_rur]
        valid = np.isfinite(sub["mean_rg"].to_numpy())
        ax_rur.plot([d for d, v in zip(week_dts, valid) if v], sub["mean_rg"].to_numpy()[valid],
                   color=RURAL_COLOR[rur], linewidth=setup_mpl.LINEWIDTH, label=rur)
    _add_boundary_lines(ax_rur)
    ax_rur.set_ylabel(r"$\langle R_g \rangle$ (km)", fontsize=setup_mpl.FONTSIZE_LABEL)
    ax_rur.legend(fontsize=setup_mpl.FONTSIZE_LEGEND, loc="lower left")

    for party in PARTY_NAMES:
        sub = df_party[df_party["stratum"] == party].set_index("week").reindex(sorted_weeks_party)
        week_dts = [_parse_week(w) for w in sorted_weeks_party]
        valid = np.isfinite(sub["mean_rg"].to_numpy())
        ax_party.plot([d for d, v in zip(week_dts, valid) if v], sub["mean_rg"].to_numpy()[valid],
                     color=PARTY_COLOR[party], linewidth=setup_mpl.LINEWIDTH, label=party)
    _add_boundary_lines(ax_party)
    ax_party.set_ylabel(r"$\langle R_g \rangle$ (km)", fontsize=setup_mpl.FONTSIZE_LABEL)
    ax_party.set_xlabel("Week", fontsize=setup_mpl.FONTSIZE_LABEL)
    ax_party.legend(fontsize=setup_mpl.FONTSIZE_LEGEND, loc="lower left")

    ax_party.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_party.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    plt.setp(ax_party.get_xticklabels(), rotation=45, ha="right", fontsize=setup_mpl.FONTSIZE_TICK)

    return fig


def plot_figure_7(csv_path=DIR_DATA / "figure_7.csv", out_dir: Path = DIR_FIGURES, fmt: str = "png") -> Path:
    """figure_7 -- weekly_rg, California."""
    return _save(draw_weekly_rg(pd.read_csv(csv_path)), out_dir, "figure_7", fmt)


# ===========================================================================
# figure_4_5_combined -- gonzalez (figure_4) stacked on top of sigmaxy
# (figure_5) in a single image, California. Does not touch draw_gonzalez /
# draw_sigmaxy: each is rendered exactly as it is for its own standalone
# figure, and the two rendered images are stitched together afterwards.
# ===========================================================================

def _fig_to_rgba(fig, dpi: float) -> np.ndarray:
    """Rasterise a matplotlib Figure to an (H, W, 4) uint8 RGBA array at *dpi*."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())


def stack_figures_vertically(figs, dpi: float = setup_mpl.DPI):
    """Stitch already-built Figures into one, stacked top to bottom.

    Each figure in *figs* is rasterised independently (unchanged) and the
    resulting images are stacked (narrower ones centre-padded to the widest)
    into a new Figure sized to match, ready to hand to ``_save``.
    """
    arrays = [_fig_to_rgba(f, dpi) for f in figs]
    for f in figs:
        plt.close(f)

    width = max(a.shape[1] for a in arrays)
    padded = []
    for a in arrays:
        pad_w = width - a.shape[1]
        if pad_w:
            left = pad_w // 2
            a = np.pad(a, ((0, 0), (left, pad_w - left), (0, 0)))
        padded.append(a)
    stacked = np.concatenate(padded, axis=0)

    fig = plt.figure(figsize=(stacked.shape[1] / dpi, stacked.shape[0] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(stacked)
    ax.axis("off")
    return fig


def plot_figure_4_5_combined(
    csv_path_4=DIR_DATA / "figure_4.csv", csv_path_5=DIR_DATA / "figure_5.csv",
    out_dir: Path = DIR_FIGURES, fmt: str = "png",
) -> Path:
    """figure_4_5_combined -- figure_4 (gonzalez) + figure_5 (sigmaxy) stacked into one image, California."""
    fig_4 = draw_gonzalez(pd.read_csv(csv_path_4))
    fig_5 = draw_sigmaxy(pd.read_csv(csv_path_5))
    fig = stack_figures_vertically([fig_4, fig_5])
    return _save(fig, out_dir, "figure_4_5_combined", fmt)


# ===========================================================================
# Registry + CLI
# ===========================================================================

FIGURE_PLOTTERS = {
    "figure_1":  plot_figure_1,
    "figure_2":  plot_figure_2,
    "figure_3":  plot_figure_3,
    "figure_4":  plot_figure_4,
    "figure_5":  plot_figure_5,
    "figure_6":  plot_figure_6,
    "figure_7":  plot_figure_7,
    "figure_8":  plot_figure_8,
    "figure_9":  plot_figure_9,
    "figure_10": plot_figure_10,
    "figure_11": plot_figure_11,
    "figure_12": plot_figure_12,
    "figure_13": plot_figure_13,
}

# Plotters that combine more than one figure_<i>.csv into a single image --
# handled separately from FIGURE_PLOTTERS since "csv_path = data_dir /
# f'{name}.csv'" doesn't apply to them. Maps name -> (plotter_fn, [csv stems]).
COMBINED_PLOTTERS = {
    "figure_4_5_combined": (plot_figure_4_5_combined, ["figure_4", "figure_5"]),
}


def run(args) -> None:
    setup_mpl.setup()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    targets  = args.only or list(FIGURE_PLOTTERS) + list(COMBINED_PLOTTERS)

    for name in targets:
        if name in COMBINED_PLOTTERS:
            plotter, deps = COMBINED_PLOTTERS[name]
            csv_paths = [data_dir / f"{d}.csv" for d in deps]
            missing = [p for p in csv_paths if not p.exists()]
            if missing:
                print(f"  SKIP {name}: {missing} not found -- run extract_figure_data.py first.")
                continue
            out_path = plotter(*csv_paths, out_dir)
            print(f"  Wrote {out_path}")
            continue

        plotter = FIGURE_PLOTTERS.get(name)
        if plotter is None:
            print(f"  Unknown figure {name!r} -- skipping "
                  f"(choices: {sorted(FIGURE_PLOTTERS) + sorted(COMBINED_PLOTTERS)}).")
            continue
        csv_path = data_dir / f"{name}.csv"
        if not csv_path.exists():
            print(f"  SKIP {name}: {csv_path} not found -- run extract_figure_data.py first.")
            continue
        out_path = plotter(csv_path, out_dir)
        print(f"  Wrote {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate figure_1.png .. figure_13.png (plus figure_4_5_combined.png) "
                     "from paper/data/figure_<i>.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--only", nargs="+", default=None, help="Subset of figures to regenerate.")
    parser.add_argument("--data-dir", default=str(DIR_DATA), help="Directory containing figure_<i>.csv.")
    parser.add_argument("--out-dir", default=str(DIR_FIGURES), help="Directory to write figure_<i>.png into.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
