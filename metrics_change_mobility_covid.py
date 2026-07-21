"""
metrics_change_mobility_covid.py
=================================
Reference implementation of every mobility metric used in the paper's
figures, stripped of all the pipeline machinery (S3 progressive download,
shard checkpointing, ``ParquetStore`` resume state, raw-shard
preprocessing) that the production ``src/`` code needs to run at scale.

This module has exactly one job: given an already-preprocessed stop-point
table for one period, compute the metric. It takes plain ``polars.DataFrame``
input and has no dependency on ``src/`` — you can copy this single file out
of the repository and it will still work, given ``numpy``/``pandas``/
``polars``/``scipy``.

Expected input schema
----------------------
Every function below takes a ``polars.DataFrame`` of stop points with (at
least) these columns:

  user_id       str/Utf8      user identifier
  lat, lon      float64       stop-point centroid (WGS-84 degrees)
  begin, end    datetime      stop start / end time
  dur_min       float64       stop duration in minutes, ``= (end - begin)``
                               in minutes, clipped at 0
  geohash7      str           geohash (precision 7) of the stop location

If you only have ``begin``/``end``, derive ``dur_min`` with::

    df = df.with_columns(
        (pl.col("end") - pl.col("begin")).dt.total_minutes()
        .cast(pl.Float64).clip(lower_bound=0.0).alias("dur_min")
    )

Rows should already be restricted to one (region, period) and to users
meeting the minimum-stops-per-period threshold (``np_``) — this module
does not do that filtering itself, since that's a data-selection concern,
not a metric formula.

Formulas below are verbatim ports of ``src/vectorized_pipeline.py`` (the
production implementation) with one deliberate exception:
``gonzalez_pca_legacy`` / ``gonzalez_pca_legacy_batch``, which reproduce
the *old* (pre-vectorization) Gonzalez PCA formula from
``src/check_gonzalez/main_computation_gonzalez_old.py`` — bugs and all
(see its docstring below) — because that is the implementation actually
used to produce the paper's gonzalez/sigmaxy figures (4/5/11/12) and the
``sigmaxy_pct_change`` sensitivity feature.

Metrics provided
-----------------
  radius_of_gyration(df)            -> [user_id, radius_gyration_km]
  k_radius_of_gyration(df, ks)      -> [user_id, rg_<k>_km, ...]
  entropies(df)                     -> [user_id, random_entropy, uncorrelated_entropy]
  real_entropy(df)                  -> [user_id, real_entropy]           (LZ78 estimator)
  lz78_entropy(locations)           -> float                              (single-user helper)
  jump_lengths(df)                  -> [user_id, step_km]                 (every inter-stop hop)
  total_distance(df)                -> [user_id, distance_km]             (sum of hops per user)
  gonzalez_pca(lat, lon)            -> (x_norm, y_norm, sigma_x_km, sigma_y_km) | None
  gonzalez_pca_batch(df)            -> [user_id, x_norm, y_norm, sigma_x_km, sigma_y_km]
  gonzalez_pca_legacy(lat, lon)     -> same signature, old/buggy formula (used for the paper figures)
  gonzalez_pca_legacy_batch(df)     -> same schema, old/buggy formula
  s_t_curve(t_hours, geohashes, t_step) -> list[int]                      (single-user helper)
  st_exploration_curves(df, t_threshold) -> {user_id: [S(0), S(1), ...]}
  frequency_by_rank(df)             -> [user_id, geohash7, frequency, rank]
  weekly_radius_of_gyration(df, week_starts, window_days=7) -> [user_id, week, radius_gyration_km]
  power_law_fit(x, y)               -> (slope, std_err, r_value, intercept)  (log-log OLS)
  log_histogram(values, bins)       -> (pdf, bin_centers, bin_widths)        (log-spaced PDF)
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as _stats

_PI    = math.pi
_R_KM  = 6371.0        # WGS-84 mean radius, km — haversine convention used throughout
K_RADIUS_VALUES_DEFAULT = [3, 6, 10]


# ---------------------------------------------------------------------------
# Radius of gyration / k-radius of gyration
# ---------------------------------------------------------------------------

def radius_of_gyration(df: pl.DataFrame) -> pl.DataFrame:
    """
    Time-weighted radius of gyration for every user in ``df``.

        rg = sqrt( sum_i( dur_i * d_i^2 ) / sum_i(dur_i) )

    where ``d_i`` is the tangent-plane distance of stop *i* from the
    user's time-weighted centroid.

    Returns ``[user_id, radius_gyration_km]``.
    """
    cm = df.group_by("user_id").agg([
        (pl.col("lat") * pl.col("dur_min")).sum() / pl.col("dur_min").sum(),
        (pl.col("lon") * pl.col("dur_min")).sum() / pl.col("dur_min").sum(),
    ]).rename({"lat": "cm_lat", "lon": "cm_lon"})

    df2 = (
        df.join(cm, on="user_id")
        .with_columns(
            (0.6e5 * (1.85533 - 0.006222 * (pl.col("cm_lat") * _PI / 180).sin())).alias("c_lat")
        )
        .with_columns((pl.col("c_lat") * (pl.col("cm_lat") * _PI / 180).cos()).alias("c_lon"))
        .with_columns((
            (pl.col("c_lon") * (pl.col("lon") - pl.col("cm_lon"))).pow(2)
            + (pl.col("c_lat") * (pl.col("lat") - pl.col("cm_lat"))).pow(2)
        ).alias("d2_m2"))
    )

    return df2.group_by("user_id").agg(
        ((pl.col("dur_min") * pl.col("d2_m2")).sum() / pl.col("dur_min").sum())
        .sqrt().alias("radius_gyration_km")
    ).with_columns((pl.col("radius_gyration_km") / 1000.0).alias("radius_gyration_km"))


def k_radius_of_gyration(df: pl.DataFrame, k_values: "list[int]" = None) -> pl.DataFrame:
    """
    k-radius of gyration: same formula as ``radius_of_gyration``, but
    computed only over each user's ``k`` most-visited locations (by visit
    count).

    Returns ``[user_id, rg_<k>_km, ...]`` for every ``k`` in ``k_values``.
    """
    k_values = k_values or K_RADIUS_VALUES_DEFAULT
    loc_counts = (
        df.group_by(["user_id", "geohash7"]).agg(pl.len().alias("n_visits"))
        .sort(["user_id", "n_visits"], descending=[False, True])
    )
    results = df.select("user_id").unique()
    for k in k_values:
        top_k = loc_counts.group_by("user_id").head(k).select(["user_id", "geohash7"])
        df_k = df.join(top_k, on=["user_id", "geohash7"], how="inner")
        if df_k.is_empty():
            results = results.with_columns(pl.lit(float("nan")).alias(f"rg_{k}_km"))
            continue
        rg_k = radius_of_gyration(df_k).rename({"radius_gyration_km": f"rg_{k}_km"})
        results = results.join(rg_k, on="user_id", how="left")
    return results


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def entropies(df: pl.DataFrame) -> pl.DataFrame:
    """
    Random entropy and uncorrelated entropy for every user.

        random_entropy        = log2(n_distinct_locations)
        uncorrelated_entropy  = -sum_i( p_i * log2(p_i) )
                                 where p_i = visits_at_i / total_visits

    Returns ``[user_id, random_entropy, uncorrelated_entropy]`` (bits).
    """
    loc_stats = df.group_by(["user_id", "geohash7"]).agg(pl.len().alias("n_visits"))
    user_totals = loc_stats.group_by("user_id").agg(
        pl.col("n_visits").sum().alias("total_visits"),
        pl.len().alias("n_locs"),
    )
    return (
        loc_stats.join(user_totals, on="user_id")
        .with_columns((pl.col("n_visits") / pl.col("total_visits")).alias("p_i"))
        .with_columns((-(pl.col("p_i") * pl.col("p_i").log(2))).alias("h_contrib"))
        .group_by("user_id")
        .agg(
            pl.col("n_locs").first().log(2).alias("random_entropy"),
            pl.col("h_contrib").sum().alias("uncorrelated_entropy"),
        )
    )


def lz78_entropy(locations: "list") -> float:
    """
    Lempel-Ziv 78 entropy-rate estimator for one ordered sequence of
    visited locations (same result as skmob's ``real_entropy``).

    Parameters
    ----------
    locations : list of hashable
        Ordered sequence of visited location identifiers (e.g. geohash7),
        one per stop, sorted by time.

    Returns
    -------
    float — estimated entropy rate in bits.
    """
    n = len(locations)
    if n <= 1:
        return 0.0
    dictionary: set = set()
    phrase: list = []
    n_phrases = 0
    for loc in locations:
        phrase.append(loc)
        key = tuple(phrase)
        if key not in dictionary:
            dictionary.add(key)
            n_phrases += 1
            phrase = []
    if phrase:
        n_phrases += 1
    if n_phrases <= 1:
        return 0.0
    return n_phrases * math.log2(n_phrases) / n


def real_entropy(df: pl.DataFrame) -> pl.DataFrame:
    """
    Real entropy (LZ78 estimator) for every user — the only inherently
    sequential metric here, one pass per user over their time-ordered
    location sequence.

    Returns ``[user_id, real_entropy]`` (bits).
    """
    def _group_fn(grp: pl.DataFrame) -> pl.DataFrame:
        uid  = grp["user_id"][0]
        locs = grp.sort("begin")["geohash7"].to_list()
        return pl.DataFrame({"user_id": [uid], "real_entropy": [lz78_entropy(locs)]})

    return df.group_by("user_id").map_groups(_group_fn)


# ---------------------------------------------------------------------------
# Distance / jump lengths
# ---------------------------------------------------------------------------

def _haversine_expr(lat_a: str, lon_a: str, lat_b: str, lon_b: str) -> pl.Expr:
    d2r = _PI / 180.0
    dlat2 = ((pl.col(lat_b) - pl.col(lat_a)) * d2r / 2).sin().pow(2)
    dlon2 = ((pl.col(lon_b) - pl.col(lon_a)) * d2r / 2).sin().pow(2)
    cos1  = (pl.col(lat_a) * d2r).cos()
    cos2  = (pl.col(lat_b) * d2r).cos()
    return 2 * _R_KM * (dlat2 + cos1 * cos2 * dlon2).sqrt().arcsin()


def jump_lengths(df: pl.DataFrame) -> pl.DataFrame:
    """
    Per-hop haversine distance [km] between consecutive stops of the same
    user — the pooled inter-stop *jump-length* distribution (figures
    1/8's right-hand panel), not the per-user total.

    Returns ``[user_id, step_km]`` — one row per hop.
    """
    df_s = df.sort(["user_id", "begin"]).with_columns([
        pl.col("lat").shift(1).over("user_id").alias("prev_lat"),
        pl.col("lon").shift(1).over("user_id").alias("prev_lon"),
        pl.col("user_id").shift(1).over("user_id").alias("prev_uid"),
    ])
    pairs = df_s.filter(pl.col("user_id") == pl.col("prev_uid")).drop("prev_uid")
    pairs = pairs.with_columns(_haversine_expr("prev_lat", "prev_lon", "lat", "lon").alias("step_km"))
    return pairs.select(["user_id", "step_km"])


def total_distance(df: pl.DataFrame) -> pl.DataFrame:
    """
    Total haversine path length [km] per user (sum of all inter-stop
    hops) — the ``all_scalars.distance`` convention, equivalent to
    ``skmob.measures.individual.distance_straight_line``.

    Returns ``[user_id, distance_km]``.
    """
    return jump_lengths(df).group_by("user_id").agg(pl.col("step_km").sum().alias("distance_km"))


# ---------------------------------------------------------------------------
# Gonzalez PCA trajectory shape
# ---------------------------------------------------------------------------

def gonzalez_pca(lat: np.ndarray, lon: np.ndarray):
    """
    Gonzalez et al. PCA trajectory-shape computation (current formula,
    ``src/vectorized_pipeline._gonzalez_pca``). Rotates each user's stops
    into their trajectory's own principal-axis frame and normalises by
    the per-axis standard deviation.

    Returns ``(x_norm, y_norm, sigma_x_km, sigma_y_km)`` or ``None`` on
    degenerate input (fewer than 2 stops).
    """
    if len(lat) < 2:
        return None
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    mean_lat, mean_lon = lat.mean(), lon.mean()
    c_lat = 0.6e5 * (1.85533 - 0.006222 * math.sin(mean_lat * _PI / 180.0))
    c_lon = c_lat * math.cos(mean_lat * _PI / 180.0)
    proj_x = c_lon * (lon - mean_lon)
    proj_y = c_lat * (lat - mean_lat)

    shifted_x = proj_x - proj_x.mean()
    shifted_y = proj_y - proj_y.mean()

    all_zero = (np.abs(shifted_x) < 1).all() and (np.abs(shifted_y) < 1).all()
    if all_zero:
        shifted_x = np.zeros_like(shifted_x)
        shifted_y = np.zeros_like(shifted_y)

    Ixx = np.sum(shifted_x ** 2)
    Iyy = np.sum(shifted_y ** 2)
    Ixy = np.sum(shifted_x * shifted_y)
    mu  = math.sqrt(max(0.0, 4 * Ixy ** 2 + Ixx ** 2 - 2 * Ixx * Iyy + Iyy ** 2))

    if all_zero:
        cos_theta = 0.0
    else:
        denom = 0.5 * Ixx - 0.5 * Iyy + 0.5 * mu
        cos_theta = 0.0 if abs(denom) < 1e-10 else -Ixy / (denom * math.sqrt(1.0 + Ixy ** 2 / denom ** 2))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta ** 2))

    if all_zero:
        rot_x, rot_y = np.zeros_like(shifted_x), np.zeros_like(shifted_y)
    else:
        rot_x = -cos_theta * shifted_x + sin_theta * shifted_y
        rot_y = -cos_theta * shifted_y - sin_theta * shifted_x

    valid = np.isfinite(rot_x) & np.isfinite(rot_y)
    rot_x, rot_y = rot_x[valid], rot_y[valid]
    if len(rot_x) == 0:
        return None

    sigma_x = math.sqrt((rot_x ** 2).mean())
    sigma_y = math.sqrt((rot_y ** 2).mean())
    if sigma_x < 1e-5:
        sigma_x = 0.0
    if sigma_y < 1e-5:
        sigma_y = 0.0

    x_norm = rot_x / sigma_x if sigma_x != 0 else np.zeros_like(rot_x)
    y_norm = rot_y / sigma_y if sigma_y != 0 else np.zeros_like(rot_y)

    # sigma is stored in metres by the projection above; convert to km.
    return x_norm, y_norm, sigma_x / 1000.0, sigma_y / 1000.0


def gonzalez_pca_batch(df: pl.DataFrame) -> pl.DataFrame:
    """``gonzalez_pca`` applied to every user. Returns [user_id, x_norm, y_norm, sigma_x_km, sigma_y_km]."""
    frames = []
    for uid, grp in df.group_by("user_id"):
        uid_str = str(uid[0] if isinstance(uid, tuple) else uid)
        pca = gonzalez_pca(grp["lat"].to_numpy(), grp["lon"].to_numpy())
        if pca is None:
            continue
        x_norm, y_norm, sx, sy = pca
        n = len(x_norm)
        frames.append(pl.DataFrame({
            "user_id": [uid_str] * n, "x_norm": x_norm, "y_norm": y_norm,
            "sigma_x_km": [sx] * n, "sigma_y_km": [sy] * n,
        }))
    if not frames:
        return pl.DataFrame({"user_id": pl.Series([], dtype=pl.Utf8), "x_norm": pl.Series([], dtype=pl.Float64),
                             "y_norm": pl.Series([], dtype=pl.Float64), "sigma_x_km": pl.Series([], dtype=pl.Float64),
                             "sigma_y_km": pl.Series([], dtype=pl.Float64)})
    return pl.concat(frames, how="vertical_relaxed")


def gonzalez_pca_legacy(lat: np.ndarray, lon: np.ndarray):
    """
    Old (pre-vectorization) Gonzalez PCA formula from
    ``src/check_gonzalez/main_computation_gonzalez_old.py`` — this is the
    implementation that actually produced the paper's gonzalez/sigmaxy
    figures (4/5/11/12) and the ``sigmaxy_pct_change`` sensitivity
    feature, kept bug-for-bug for reproducibility:

    * the tangent-plane reference point's lat/lon are swapped
      (``xy(lat, lon, lat0=mean(lon), lon0=mean(lat))``);
    * the "all points effectively coincide" degeneracy check omits
      ``abs()`` (``(shifted < 1).all()`` instead of
      ``(abs(shifted) < 1).all()``);
    * ``cos_theta`` is not clipped to ``[-1, 1]``, so ``sin_theta`` can
      legitimately come out ``NaN`` for some users.

    Returns ``(x_norm, y_norm, sigma_x_km, sigma_y_km)`` or ``None`` on
    degenerate input.
    """
    if len(lat) < 2:
        return None
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Bug: reference point's lat/lon are swapped.
    mean_lon_swapped = lat.mean()
    mean_lat_swapped = lon.mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        c_lat = 0.6e5 * (1.85533 - 0.006222 * math.sin(mean_lat_swapped * _PI / 180.0))
        c_lon = c_lat * math.cos(mean_lat_swapped * _PI / 180.0)
        proj_x = c_lon * (lon - mean_lon_swapped)
        proj_y = c_lat * (lat - mean_lat_swapped)
        shifted_lat = proj_y - proj_y.mean()
        shifted_lng = proj_x - proj_x.mean()

        # Bug: no abs() -- only triggers when every shift is < 1, not when close to 0.
        if bool((shifted_lat < 1).all()) and bool((shifted_lng < 1).all()):
            shifted_lat = np.zeros_like(shifted_lat)
            shifted_lng = np.zeros_like(shifted_lng)

        Ixx = np.sum(shifted_lng ** 2)
        Iyy = np.sum(shifted_lat ** 2)
        Ixy = np.sum(shifted_lat * shifted_lng)
        mu  = np.sqrt(4 * Ixy ** 2 + Ixx ** 2 - 2 * Ixx * Iyy + Iyy ** 2)

        if bool((shifted_lat == 0).all()) and bool((shifted_lng == 0).all()):
            cos_theta = 0.0
        else:
            denom = 0.5 * Ixx - 0.5 * Iyy + 0.5 * mu
            cos_theta = -Ixy / (denom * np.sqrt(1 + Ixy ** 2 / denom ** 2))

        # Bug: no clipping -- sin_theta can legitimately come out NaN.
        sin_theta = np.sqrt(1 - cos_theta ** 2)

        if bool((shifted_lat < 1).all()) and bool((shifted_lng < 1).all()):
            rotated_lat = np.zeros_like(shifted_lat)
            rotated_lng = np.zeros_like(shifted_lng)
        else:
            rotated_lat = -cos_theta * shifted_lat + sin_theta * shifted_lng
            rotated_lng = -cos_theta * shifted_lng - sin_theta * shifted_lat

    valid = np.isfinite(rotated_lat) & np.isfinite(rotated_lng)
    rotated_lat, rotated_lng = rotated_lat[valid], rotated_lng[valid]
    if len(rotated_lat) == 0:
        return None

    sigma_lat = float(np.sqrt((rotated_lat ** 2).mean()))
    sigma_lng = float(np.sqrt((rotated_lng ** 2).mean()))
    if sigma_lat < 1e-5:
        sigma_lat = 0.0
    if sigma_lng < 1e-5:
        sigma_lng = 0.0

    norm_rot_lat = rotated_lat / sigma_lat if sigma_lat != 0 else np.zeros_like(rotated_lat)
    norm_rot_lng = rotated_lng / sigma_lng if sigma_lng != 0 else np.zeros_like(rotated_lng)

    # Old code's own naming: x_norm/sigmax come from the *longitude* axis,
    # y_norm/sigmay from the *latitude* axis. sigma is in metres above;
    # convert to km to match gonzalez_pca's return convention.
    return norm_rot_lng, norm_rot_lat, sigma_lng / 1000.0, sigma_lat / 1000.0


def gonzalez_pca_legacy_batch(df: pl.DataFrame) -> pl.DataFrame:
    """``gonzalez_pca_legacy`` applied to every user. Returns [user_id, x_norm, y_norm, sigma_x_km, sigma_y_km]."""
    frames = []
    for uid, grp in df.group_by("user_id"):
        uid_str = str(uid[0] if isinstance(uid, tuple) else uid)
        pca = gonzalez_pca_legacy(grp["lat"].to_numpy(), grp["lon"].to_numpy())
        if pca is None:
            continue
        x_norm, y_norm, sx, sy = pca
        n = len(x_norm)
        frames.append(pl.DataFrame({
            "user_id": [uid_str] * n, "x_norm": x_norm, "y_norm": y_norm,
            "sigma_x_km": [sx] * n, "sigma_y_km": [sy] * n,
        }))
    if not frames:
        return pl.DataFrame({"user_id": pl.Series([], dtype=pl.Utf8), "x_norm": pl.Series([], dtype=pl.Float64),
                             "y_norm": pl.Series([], dtype=pl.Float64), "sigma_x_km": pl.Series([], dtype=pl.Float64),
                             "sigma_y_km": pl.Series([], dtype=pl.Float64)})
    return pl.concat(frames, how="vertical_relaxed")


# ---------------------------------------------------------------------------
# S(t) exploration curve
# ---------------------------------------------------------------------------

def s_t_curve(t_hours: np.ndarray, geohashes: "list", t_step: int, n_steps: int) -> "list[int]":
    """
    S(t) exploration curve for one user: number of *distinct* locations
    visited by hour ``t``, for ``t = 0, t_step, 2*t_step, ..., < n_steps``
    -- actually evaluated at every integer hour 0..n_steps-1 (``t_step``
    only affects how densely the caller chooses to sample it downstream;
    this returns the full per-hour curve).

    Parameters
    ----------
    t_hours : int64 array, sorted ascending, first entry 0
        Hour offset of each stop from the user's first stop.
    geohashes : list[str]
        Geohash7 of each stop, same order as ``t_hours``.
    t_step : int
        Unused here (kept for API symmetry with the burst-filter
        t_threshold) -- present for callers that want it, e.g. to
        subsample ``result[::t_step]``.
    n_steps : int
        Length of the returned curve (e.g. the observation window in
        hours). ``result[h]`` = distinct locations seen by hour ``h``.

    Returns
    -------
    list[int], length ``n_steps``.
    """
    n = len(t_hours)
    if n == 0:
        return [0] * n_steps

    visited: set = set()
    s_arr = np.empty(n, dtype=np.int32)
    for i, gh in enumerate(geohashes):
        visited.add(gh)
        s_arr[i] = len(visited)

    t_arr = np.asarray(t_hours, dtype=np.int64)
    result = [0] * n_steps
    e = 0
    for h in range(n_steps):
        while e < n - 1 and t_arr[e + 1] <= h:
            e += 1
        if t_arr[e] <= h:
            result[h] = int(s_arr[e])
    return result


def st_exploration_curves(df: pl.DataFrame, n_steps: int, t_threshold: int = 1) -> "dict[str, list]":
    """``s_t_curve`` applied to every user. Returns ``{user_id: [S(0), S(1), ...]}`` of length ``n_steps``."""
    result: dict = {}
    for uid, grp in df.sort("begin").group_by("user_id"):
        uid_str = str(uid[0] if isinstance(uid, tuple) else uid)
        t0 = grp["begin"][0]
        t_hours = (grp["begin"] - t0).dt.total_hours().to_numpy().astype(np.int64)
        result[uid_str] = s_t_curve(t_hours, grp["geohash7"].to_list(), t_threshold, n_steps)
    return result


# ---------------------------------------------------------------------------
# Frequency / rank
# ---------------------------------------------------------------------------

def frequency_by_rank(df: pl.DataFrame) -> pl.DataFrame:
    """
    Visit frequency and rank per (user, location): ``rank=1`` is a user's
    most-visited location, ``frequency`` is that location's share of the
    user's total visits.

    Returns ``[user_id, geohash7, frequency, rank]``.
    """
    loc_counts = (
        df.group_by(["user_id", "geohash7"]).agg(pl.len().alias("n_visits"))
        .sort(["user_id", "n_visits"], descending=[False, True])
    )
    user_totals = loc_counts.group_by("user_id").agg(pl.col("n_visits").sum().alias("total"))
    loc_counts = loc_counts.join(user_totals, on="user_id").with_columns(
        (pl.col("n_visits") / pl.col("total")).alias("frequency")
    )
    return (
        loc_counts.sort(["user_id", "n_visits"], descending=[False, True])
        .with_columns(pl.col("n_visits").cum_count().over("user_id").alias("rank"))
        .select(["user_id", "geohash7", "frequency", "rank"])
    )


# ---------------------------------------------------------------------------
# Weekly radius of gyration
# ---------------------------------------------------------------------------

def weekly_radius_of_gyration(
    df: pl.DataFrame, week_starts: "Iterable", window_days: int = 7,
) -> pl.DataFrame:
    """
    Radius of gyration computed independently within each rolling
    ``window_days``-day window starting at each timestamp in
    ``week_starts`` -- the weekly RG time series behind figure_7.

    Parameters
    ----------
    df : pl.DataFrame
        Stop points for one region/period (any span covering the weeks
        of interest), with the standard schema (module docstring).
    week_starts : iterable of datetime
        Left edge of each weekly window.
    window_days : int
        Window length in days (default 7).

    Returns
    -------
    pl.DataFrame — ``[user_id, week, radius_gyration_km]``, ``week`` is
    the ISO string of that window's start timestamp.
    """
    from datetime import timedelta

    frames = []
    for week_start in week_starts:
        week_end = week_start + timedelta(days=window_days)
        week_df = df.filter((pl.col("begin") >= pl.lit(week_start)) & (pl.col("begin") < pl.lit(week_end)))
        if week_df.is_empty():
            continue
        rg = radius_of_gyration(week_df).with_columns(pl.lit(str(week_start)).alias("week"))
        frames.append(rg)
    if not frames:
        return pl.DataFrame({"user_id": pl.Series([], dtype=pl.Utf8),
                             "radius_gyration_km": pl.Series([], dtype=pl.Float64),
                             "week": pl.Series([], dtype=pl.Utf8)})
    return pl.concat(frames, how="vertical_relaxed").select(["user_id", "week", "radius_gyration_km"])


# ---------------------------------------------------------------------------
# Small statistical helpers used by the plots
# ---------------------------------------------------------------------------

def power_law_fit(x, y):
    """
    OLS fit of ``log(y) ~ log(x)`` -- the S(t) exploration exponent mu
    (``S(t) ~ t^mu``) and the sensitivity-analysis ``st_exponent_mu``
    feature both use this. Only points with ``x > 0`` and ``y > 0`` are
    used.

    Returns ``(slope, std_err, r_value, intercept_of_the_fit_in_original_units)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0)
    slope, intercept, r_value, _p_value, std_err = _stats.linregress(np.log(x[mask]), np.log(y[mask]))
    return slope, std_err, r_value, np.exp(intercept)


def log_histogram(values, bins: int):
    """
    Log-spaced probability density: histograms ``log(values)`` on
    ``bins`` linear bins, then maps back to linear space and normalises
    by each bin's linear width so the result integrates to 1 over the
    original (non-log) values -- the log-log PDF convention used by
    every ``check_*`` script (``rg_histograms.logHist``).

    Only strictly positive values are used.

    Returns ``(pdf, bin_centers, bin_widths)``, all ``np.ndarray``.
    """
    x = np.asarray(values, dtype=float)
    x = x[x > 0]
    counts, log_edges = np.histogram(np.log(x), bins)
    edges = np.exp(log_edges)
    widths  = edges[1:] - edges[:-1]
    centers = 0.5 * (edges[1:] + edges[:-1])
    total = counts.sum()
    pdf = counts / total / widths if total > 0 else np.zeros_like(counts, dtype=float)
    return pdf, centers, widths
