"""
extract_figure_data.py
=======================
Transforms each consolidated ``check_*``/``distance_check`` result tree
under ``milestones_analysis/<region>/...`` (and the accumulated
``milestones_analysis/sensitivity_analysis.json``) into the flat
``figure_<i>.csv`` files this ``paper/`` package ships, plus the
``<feature>_t_threshold_<8|24>_MA-CA.csv`` sensitivity tables.

This script MUST run in an environment that has access to the actual
consolidated parquet trees — i.e. on the server, after:

  * ``main_computation_rg_krg.py`` / ``main_computation_entropy.py`` /
    ``main_computation_frequency_st.py`` / ``main_computation_distance_new.py`` /
    ``main_computation_gonzalez_old.py`` / ``main_computation_rg_weekly.py``
    have been run for ``--region CA --np 20 --t-threshold 1`` and
    ``--region MA --np 20 --t-threshold 1`` (rg_weekly is only needed for CA
    — see figure_7 below);
  * ``sensitivity_analysis.py`` has been run for ``--t-threshold 8`` and
    ``--t-threshold 24`` (both regions), if you want the sensitivity CSVs.

It does not recompute anything — it reuses the exact ``build_*``
data-preparation functions already used by each ``check_*/plot_*.py``
script (imported from ``src/``), so every row in ``figure_<i>.csv``
matches what the corresponding paper figure actually plots. This is the
*only* script in this package that touches ``src/`` or needs the raw
consolidated data — ``plot_change_mobility_covid.py`` and
``metrics_change_mobility_covid.py`` only ever read the CSVs this script
writes.

Figures (all at t_threshold = 1, np = 20)
------------------------------------------
  figure_1  / figure_8   rg_and_dist   (CA / MA)
  figure_2  / figure_9   rg_krg        (CA / MA)
  figure_3  / figure_10  st_and_frequency (CA / MA)
  figure_4  / figure_11  gonzalez      (CA / MA)
  figure_5  / figure_12  sigmaxy       (CA / MA)
  figure_6  / figure_13  entropy       (CA / MA)
  figure_7                weekly_rg    (CA only)

Column contracts are documented at the top of each ``build_figure_*``
function below and in ``paper/README.md``.

Usage
-----
  python paper/extract_figure_data.py
  python paper/extract_figure_data.py --only figure_1 figure_8 sensitivity
  python paper/extract_figure_data.py --np 20 --t-threshold 1
  python paper/extract_figure_data.py --only sensitivity --sensitivity-t-thresholds 8 24

Output
------
  paper/data/figure_1.csv  ...  paper/data/figure_13.csv
  paper/data/<feature>_t_threshold_8_MA-CA.csv
  paper/data/<feature>_t_threshold_24_MA-CA.csv
"""

import argparse
import json
import sys
from pathlib import Path

# ─── Make the src package importable regardless of working directory ──────
DIR_PAPER    = Path(__file__).resolve().parent          # HumMobCov/paper
PROJECT_ROOT = DIR_PAPER.parent                          # HumMobCov/
DIR_SRC      = PROJECT_ROOT / "src"
for _p in (PROJECT_ROOT, DIR_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DIR_DATA = DIR_PAPER / "data"

import numpy as np
import pandas as pd

from src.constants import (
    DIR_MILESTONES_SERVER, PERIOD_NAMES, MIN_POINTS_PER_USER, TIME_THRESHOLD_HOURS,
)

REGIONS = ["MA"] #, "CA"]


def _diagnose_base(label: str, base: Path) -> None:
    """
    Print exactly which directory a build_figure_* function is about to
    read, whether it exists, and (if it does) what's directly inside it
    and each period subdirectory — so a "no data found" result is
    self-explanatory instead of a silent empty CSV. Cheap (one os.listdir
    per call), always safe to call even if base doesn't exist.
    """
    print(f"    [{label}] checking {base}  (exists: {base.exists()})")
    if not base.exists():
        return
    for child in sorted(base.iterdir()):
        if child.is_dir():
            grandchildren = sorted(p.name for p in child.iterdir()) if child.exists() else []
            print(f"      {child.name}/  ->  {grandchildren}")
        else:
            print(f"      {child.name}")


# ---------------------------------------------------------------------------
# figure_1 / figure_8 — rg_and_dist
#   columns: panel [radius_of_gyration|jump_length], period, bin_center, pdf,
#            fit_alpha, fit_sigma, fit_xmin, fit_fraction
#   (fit_* only populated for the jump_length panel)
# ---------------------------------------------------------------------------

def build_figure_rg_and_dist(region: str, np_: int, t_threshold: int) -> pd.DataFrame:
    from src.distance_check.visualiza_rg_and_dist_distribution import build_rg_dataframe
    from src.distance_check.plot_distance_jumps import (
        build_dataframe as build_distance_dataframe,
        DEFAULT_CUT_KM, DEFAULT_XMIN_SEARCH_MIN, DEFAULT_XMIN_SEARCH_MAX,
        DEFAULT_MAX_FIT_SAMPLES,
    )

    cols = ["panel", "period", "bin_center", "pdf",
            "fit_alpha", "fit_sigma", "fit_xmin", "fit_fraction"]

    # The rg panel here is read from the rg_krg tree (main_computation_rg_krg.py's
    # output), NOT from distance_jumps -- a common cause of "rg panel is empty"
    # is that rg_krg was never computed even though distance_jumps was.
    _diagnose_base("rg_and_dist:rg", DIR_MILESTONES_SERVER / region / "rg_krg" / f"np_{np_}_t_{t_threshold}")
    rg_df = build_rg_dataframe(region, np_, t_threshold)
    if not rg_df.empty:
        rg_df = rg_df.assign(panel="radius_of_gyration", fit_alpha=np.nan,
                              fit_sigma=np.nan, fit_xmin=np.nan, fit_fraction=np.nan)

    # No upper cut on jump length here — same default as the actual combined
    # figure (visualiza_rg_and_dist_distribution.py's --cut-max-km=inf), not
    # the 1000 km default of the standalone distance-only plot.
    dist_df = build_distance_dataframe(
        region, np_, t_threshold, DEFAULT_CUT_KM, None,
        DEFAULT_XMIN_SEARCH_MIN, DEFAULT_XMIN_SEARCH_MAX, DEFAULT_MAX_FIT_SAMPLES,
        float("inf"),
    )
    if not dist_df.empty:
        dist_df = dist_df.assign(panel="jump_length")

    parts = [d for d in (rg_df, dist_df) if not d.empty]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols]


# ---------------------------------------------------------------------------
# figure_2 / figure_9 — rg_krg
#   columns: period, k, rg_km, krg_km, density   (zero-density cells omitted)
# ---------------------------------------------------------------------------

def build_figure_rg_krg(region: str, np_: int, t_threshold: int) -> pd.DataFrame:
    from src.check_rg_krg.plot_rg_krg import build_histograms

    _diagnose_base("rg_krg", DIR_MILESTONES_SERVER / region / "rg_krg" / f"np_{np_}_t_{t_threshold}")
    Hs, centers = build_histograms(region, np_, t_threshold)
    rows = []
    for (period, k), H in Hs.items():
        # H is [row=krg bin, col=rg bin] (plot_rg_krg.build_histograms convention).
        nz_rows, nz_cols = np.nonzero(H)
        for i, j in zip(nz_rows, nz_cols):
            rows.append({
                "period": period, "k": int(k),
                "rg_km": float(centers[j]), "krg_km": float(centers[i]),
                "density": float(H[i, j]),
            })
    return pd.DataFrame(rows, columns=["period", "k", "rg_km", "krg_km", "density"])


# ---------------------------------------------------------------------------
# figure_3 / figure_10 — st_and_frequency
#   columns: panel [st|frequency], period, time_step, mean_st, std_st,
#            rank, mean_frequency
# ---------------------------------------------------------------------------

def build_figure_st_and_frequency(region: str, np_: int, t_threshold: int) -> pd.DataFrame:
    from src.check_frequency_st.plot_frequency_st import (
        build_st_dataframe, build_frequency_dataframe, DEFAULT_MAX_RANK,
    )

    cols = ["panel", "period", "time_step", "mean_st", "std_st", "rank", "mean_frequency"]

    st_df = build_st_dataframe(region, np_, t_threshold)
    if not st_df.empty:
        st_df = st_df.assign(panel="st", rank=np.nan, mean_frequency=np.nan)

    freq_df = build_frequency_dataframe(region, np_, t_threshold, DEFAULT_MAX_RANK)
    if not freq_df.empty:
        freq_df = freq_df.assign(panel="frequency", time_step=np.nan,
                                  mean_st=np.nan, std_st=np.nan)

    parts = [d for d in (st_df, freq_df) if not d.empty]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)[cols]


# ---------------------------------------------------------------------------
# figure_4 / figure_11 — gonzalez (2-D density in the intrinsic trajectory
#   reference frame)
#   columns: period, x_norm, y_norm, count   (zero-count cells omitted)
# ---------------------------------------------------------------------------

def build_figure_gonzalez(region: str, np_: int, t_threshold: int) -> pd.DataFrame:
    from src.check_gonzalez.plot_gonzalez_old import _iter_period_chunks, DEFAULT_BATCH_SIZE

    base = DIR_MILESTONES_SERVER / region / "gonzalez_old" / f"np_{np_}_t_{t_threshold}"
    xmin, xmax, ymin, ymax, nbins = -1.5, 1.5, -2.2, 2.2, 40   # same as plot_gonzalez_old.plot_gonzalez_old
    x_edges   = np.linspace(xmin, xmax, nbins + 1)
    y_edges   = np.linspace(ymin, ymax, nbins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    rows = []
    for p in PERIOD_NAMES:
        H = np.zeros((nbins, nbins))
        got_data = False
        for chunk in _iter_period_chunks(base, p, ["x_norm", "y_norm"], DEFAULT_BATCH_SIZE):
            x = chunk["x_norm"].to_numpy()
            y = chunk["y_norm"].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.any():
                continue
            h, _, _ = np.histogram2d(x[mask], y[mask], bins=(nbins, nbins),
                                     range=[[xmin, xmax], [ymin, ymax]])
            H += h.T   # shape (nbins_y, nbins_x) -- same orientation as plot_gonzalez_old
            got_data = True
        if not got_data:
            continue
        nz_rows, nz_cols = np.nonzero(H)
        for i, j in zip(nz_rows, nz_cols):
            rows.append({
                "period": p, "x_norm": float(x_centers[j]), "y_norm": float(y_centers[i]),
                "count": float(H[i, j]),
            })
    return pd.DataFrame(rows, columns=["period", "x_norm", "y_norm", "count"])


# ---------------------------------------------------------------------------
# figure_5 / figure_12 — sigmaxy
#   columns: period, axis [sigma_x|sigma_y], bin_center, pdf, mean_km,
#            mean_pct_change_vs_baseline
# ---------------------------------------------------------------------------

def build_figure_sigmaxy(region: str, np_: int, t_threshold: int) -> pd.DataFrame:
    from src.check_gonzalez.plot_gonzalez_old import _chunked_log_hist, DEFAULT_BATCH_SIZE

    base = DIR_MILESTONES_SERVER / region / "gonzalez_old" / f"np_{np_}_t_{t_threshold}"

    rows = []
    means: dict[str, dict[str, float]] = {}
    for p in PERIOD_NAMES:
        hist = _chunked_log_hist(base, p, ["sigmax", "sigmay"], bins=150, threshold=0.1,
                                  scale=1e-3, batch_size=DEFAULT_BATCH_SIZE)
        means[p] = {}
        for axis, col in [("sigma_x", "sigmax"), ("sigma_y", "sigmay")]:
            w, q, mean_c = hist[col]
            means[p][axis] = mean_c
            for bc, pv in zip(q, w):
                rows.append({"period": p, "axis": axis, "bin_center": float(bc),
                             "pdf": float(pv), "mean_km": float(mean_c)})

    df = pd.DataFrame(rows, columns=["period", "axis", "bin_center", "pdf", "mean_km"])
    if df.empty:
        df["mean_pct_change_vs_baseline"] = pd.Series(dtype=float)
        return df

    periods_present = [p for p in PERIOD_NAMES if p in means]
    baseline = periods_present[0]

    def _pct_change(row) -> float:
        if row["period"] == baseline:
            return float("nan")
        base_v = means.get(baseline, {}).get(row["axis"])
        cur_v  = row["mean_km"]
        if not base_v or not np.isfinite(base_v) or base_v <= 0 or not np.isfinite(cur_v):
            return float("nan")
        return 100.0 * (cur_v - base_v) / base_v

    df["mean_pct_change_vs_baseline"] = df.apply(_pct_change, axis=1)
    return df


# ---------------------------------------------------------------------------
# figure_6 / figure_13 — entropy
#   columns: period, entropy_type, bin_center, pdf
# ---------------------------------------------------------------------------

def build_figure_entropy(region: str, np_: int, t_threshold: int) -> pd.DataFrame:
    from src.check_entropy.plot_entropy_new import build_raw_entropy, build_histogram_dataframe

    _diagnose_base("entropy", DIR_MILESTONES_SERVER / region / "entropy" / f"np_{np_}_t_{t_threshold}")
    raw = build_raw_entropy(region, np_, t_threshold)
    if not raw:
        return pd.DataFrame(columns=["period", "entropy_type", "bin_center", "pdf"])
    return build_histogram_dataframe(raw)


# ---------------------------------------------------------------------------
# figure_7 — weekly_rg (CA only)
#   columns: panel [rurality|party], week, stratum, mean_rg, std_rg
# ---------------------------------------------------------------------------

def build_figure_weekly_rg(region: str, np_: int, t_threshold: int) -> pd.DataFrame:
    from src.check_rg_weekly.plot_weekly_rg import build_weekly_rg_stratified, build_dataframes

    _diagnose_base("rg_weekly", DIR_MILESTONES_SERVER / region / "rg_weekly" / f"np_{np_}_t_{t_threshold}")
    rg_rural, rg_party = build_weekly_rg_stratified(region, np_, t_threshold)
    df_rur, df_party = build_dataframes(rg_rural, rg_party)

    cols = ["panel", "week", "stratum", "mean_rg", "std_rg"]
    parts = []
    if not df_rur.empty:
        parts.append(df_rur.assign(panel="rurality", stratum=df_rur["rurality"])[cols])
    if not df_party.empty:
        parts.append(df_party.assign(panel="party", stratum=df_party["party"])[cols])
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Figure registry
# ---------------------------------------------------------------------------

FIGURE_BUILDERS = {
    "figure_1":  lambda np_, t: build_figure_rg_and_dist("CA", np_, t),
    "figure_2":  lambda np_, t: build_figure_rg_krg("CA", np_, t),
    "figure_3":  lambda np_, t: build_figure_st_and_frequency("CA", np_, t),
    "figure_4":  lambda np_, t: build_figure_gonzalez("CA", np_, t),
    "figure_5":  lambda np_, t: build_figure_sigmaxy("CA", np_, t),
    "figure_6":  lambda np_, t: build_figure_entropy("CA", np_, t),
    "figure_7":  lambda np_, t: build_figure_weekly_rg("CA", np_, t),
    "figure_8":  lambda np_, t: build_figure_rg_and_dist("MA", np_, t),
    "figure_9":  lambda np_, t: build_figure_rg_krg("MA", np_, t),
    "figure_10": lambda np_, t: build_figure_st_and_frequency("MA", np_, t),
    "figure_11": lambda np_, t: build_figure_gonzalez("MA", np_, t),
    "figure_12": lambda np_, t: build_figure_sigmaxy("MA", np_, t),
    "figure_13": lambda np_, t: build_figure_entropy("MA", np_, t),
}


# ---------------------------------------------------------------------------
# Sensitivity tables — flatten milestones_analysis/sensitivity_analysis.json
# (built up by src/sensitivity_analysis.py) into one MA+CA CSV per
# (feature, t_threshold). This part needs only the JSON, not the raw
# consolidated parquet trees.
# ---------------------------------------------------------------------------

def _load_sensitivity_json() -> dict:
    path = DIR_MILESTONES_SERVER / "sensitivity_analysis.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _flatten_distance_powerlaw(data: dict, t: int) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        feat = data.get(region, {}).get(str(t), {}).get("distance_powerlaw", {})
        for period, v in feat.items():
            rows.append({"region": region, "period": period, **v})
    return pd.DataFrame(rows)


def _flatten_entropy_peak(data: dict, t: int) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        feat = data.get(region, {}).get(str(t), {}).get("entropy_peak", {})
        for period, etypes in feat.items():
            for etype, v in etypes.items():
                rows.append({"region": region, "period": period, "entropy_type": etype, **v})
    return pd.DataFrame(rows)


def _flatten_frequency_top2(data: dict, t: int) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        feat = data.get(region, {}).get(str(t), {}).get("frequency_top2_pct", {})
        for period, v in feat.items():
            rows.append({"region": region, "period": period, **v})
    return pd.DataFrame(rows)


def _flatten_rg_distance_crossing(data: dict, t: int) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        feat = data.get(region, {}).get(str(t), {}).get("rg_distance_crossing", {})
        if not feat:
            continue
        for comparison, period_key in [("vs_pre", "vs_pre_period"), ("vs_post", "vs_post_period")]:
            c = feat.get(comparison, {}) or {}
            rows.append({
                "region": region, "restriction_period": feat.get("restriction_period"),
                "comparison": comparison, "comparison_period": feat.get(period_key),
                "r_g_star_km": c.get("r_g_star_km"), "area_below_km": c.get("area_below_km"),
                "area_above_km": c.get("area_above_km"), "note": c.get("note"),
            })
    return pd.DataFrame(rows)


def _flatten_rg_krg_bin0_percentile(data: dict, t: int) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        feat = data.get(region, {}).get(str(t), {}).get("rg_krg_bin0_percentile", {})
        for period, ks in feat.items():
            for k, v in ks.items():
                rows.append({"region": region, "period": period, "k": int(k), **v})
    return pd.DataFrame(rows)


def _flatten_sigmaxy_pct_change(data: dict, t: int) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        feat = data.get(region, {}).get(str(t), {}).get("sigmaxy_pct_change", {})
        for period, v in feat.items():
            rows.append({"region": region, "period": period, **v})
    return pd.DataFrame(rows)


def _flatten_st_exponent_mu(data: dict, t: int) -> pd.DataFrame:
    rows = []
    for region in REGIONS:
        feat = data.get(region, {}).get(str(t), {}).get("st_exponent_mu", {})
        for period, v in feat.items():
            rows.append({"region": region, "period": period, **v})
    return pd.DataFrame(rows)


SENSITIVITY_FEATURES = {
    "distance_powerlaw":      _flatten_distance_powerlaw,
    "entropy_peak":           _flatten_entropy_peak,
    "frequency_top2_pct":     _flatten_frequency_top2,
    "rg_distance_crossing":   _flatten_rg_distance_crossing,
    "rg_krg_bin0_percentile": _flatten_rg_krg_bin0_percentile,
    "sigmaxy_pct_change":     _flatten_sigmaxy_pct_change,
    "st_exponent_mu":         _flatten_st_exponent_mu,
}


def build_sensitivity_tables(t_thresholds: "list[int]") -> "dict[str, pd.DataFrame]":
    data = _load_sensitivity_json()
    out = {}
    for t in t_thresholds:
        for name, fn in SENSITIVITY_FEATURES.items():
            out[f"{name}_t_threshold_{t}_MA-CA"] = fn(data, t)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(args) -> None:
    DIR_DATA.mkdir(parents=True, exist_ok=True)
    targets = args.only or (list(FIGURE_BUILDERS) + ["sensitivity"])

    for name in targets:
        if name == "sensitivity":
            continue
        builder = FIGURE_BUILDERS.get(name)
        if builder is None:
            print(f"  Unknown target {name!r} -- skipping (choices: "
                  f"{sorted(FIGURE_BUILDERS)} + ['sensitivity']).")
            continue
        try:
            df = builder(args.np_, args.t_threshold)
        except FileNotFoundError as exc:
            print(f"  SKIP {name}: {exc}")
            continue
        except Exception as exc:
            print(f"  WARNING: {name} failed: {exc}")
            continue
        if df is None or df.empty:
            print(f"  {name}: no data found under {DIR_MILESTONES_SERVER} -- skipping. "
                  f"Run the matching main_computation_*.py first.")
            continue
        out_path = DIR_DATA / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Wrote {out_path}  ({len(df):,} rows)")

    if "sensitivity" in targets:
        tables = build_sensitivity_tables(args.sensitivity_t_thresholds)
        for fname, df in tables.items():
            if df.empty:
                print(f"  {fname}: no data in sensitivity_analysis.json -- skipping. "
                      f"Run src/sensitivity_analysis.py for that t_threshold first.")
                continue
            out_path = DIR_DATA / f"{fname}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Wrote {out_path}  ({len(df):,} rows)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Transform consolidated check_*/distance_check results and "
            "sensitivity_analysis.json into the flat figure_<i>.csv / "
            "<feature>_t_threshold_<t>_MA-CA.csv files shipped in paper/data/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Subset of targets to (re)build, e.g. 'figure_1 figure_8 sensitivity'. "
             "Default: every figure_1..figure_13 plus sensitivity.",
    )
    parser.add_argument(
        "--np", dest="np_", type=int, default=MIN_POINTS_PER_USER,
        help="Minimum stops per user per period (must match an already-computed case).",
    )
    parser.add_argument(
        "--t-threshold", dest="t_threshold", type=int, default=TIME_THRESHOLD_HOURS,
        help="Minimum hours between stops for figure_1..figure_13 (the paper figures "
             "are all at t_threshold=1).",
    )
    parser.add_argument(
        "--sensitivity-t-thresholds", nargs="+", type=int, default=[8, 24],
        help="t_threshold values to render into <feature>_t_threshold_<t>_MA-CA.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
