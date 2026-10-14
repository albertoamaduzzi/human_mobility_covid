# Statistical laws of human mobility under COVID-19 restrictions

This repository reproduces the metrics and figures of the paper comparing human mobility
patterns in **California** and **Massachusetts** before, during, and after the 2020 COVID-19
restriction period. Each region's observation window is split into three periods — pre-restriction
(15 Jan – 15 Mar), restriction (15 Mar – 15 May), and post-restriction (15 May – 30 Sep) — and the
figures compare mobility behaviour across the three.

It ships three things:

* the pre-computed, aggregated data behind every figure (`data/`);
* the plotting code that turns that data into the published figures (`plot_change_mobility_covid.py`
  / `.ipynb`);
* a documented, from-scratch reference implementation of every mobility metric used in the paper
  (`metrics_change_mobility_covid.py`).

## Layout

```
README.md                          this file
setup_mpl.py                       shared matplotlib/seaborn styling (fonts, colours, DPI)
metrics_change_mobility_covid.py   reference implementation of every metric (see "Methods" below)
plot_change_mobility_covid.py      every figure (figure_1 .. figure_13, plus figure_4_5_combined)
plot_change_mobility_covid.ipynb   notebook that reproduces the whole figure set, cell by cell
extract_figure_data.py             documents how data/figure_<i>.csv was originally produced from
                                    raw mobility data (see "Where the data comes from" below)
data/
  figure_1.csv  ...  figure_13.csv             one row per plotted data point (see below)
  <feature>_t_threshold_{8,24}_MA-CA.csv       robustness/sensitivity tables (see below)
figures/
  figure_1.png  ...  figure_13.png, figure_4_5_combined.png     pre-rendered outputs
```

## Reproducing the figures

1. Install the dependencies: `pip install numpy pandas matplotlib seaborn scipy polars`.
2. Either:
   * open `plot_change_mobility_covid.ipynb` and run all cells — each cell reads the matching
     `data/figure_<i>.csv`, redraws the figure, saves it to `figures/figure_<i>.png`, and displays
     it inline; or
   * run from the command line, from this directory:
     ```
     python plot_change_mobility_covid.py                            # every figure
     python plot_change_mobility_covid.py --only figure_1 figure_8   # a subset
     ```

Every figure regenerates entirely from the CSVs already committed under `data/` — no other input,
server, or external service is needed.

## Figures

| # | Content | Region |
|---|---|---|
| 1  | Radius of gyration (left) & jump-length distribution (right) | California |
| 2  | k-radius of gyration vs. radius of gyration, per period x k | California |
| 3  | S(t) exploration curve (left) & location frequency by rank (right) | California |
| 4  | Gonzalez trajectory-shape 2-D density + profile | California |
| 5  | sigma_x / sigma_y distributions | California |
| 4+5 | `figure_4_5_combined` — figure_4 and figure_5 stacked into a single image | California |
| 6  | Entropy distributions (random / uncorrelated / real / scaled) | California |
| 7  | Weekly radius of gyration, by rurality & political party | California |
| 8  | Radius of gyration & jump-length distribution | Massachusetts |
| 9  | k-radius of gyration vs. radius of gyration | Massachusetts |
| 10 | S(t) exploration curve & location frequency by rank | Massachusetts |
| 11 | Gonzalez trajectory-shape 2-D density + profile | Massachusetts |
| 12 | sigma_x / sigma_y distributions | Massachusetts |
| 13 | Entropy distributions | Massachusetts |

There is no weekly-RG figure for Massachusetts (rurality/party metadata was only available for
California). All figures require a minimum of 20 stops per user per period (`np = 20`) and use a
1-hour burst-filter threshold (`t_threshold = 1`) — see the sensitivity tables below for how the
results hold at the coarser 8- and 24-hour thresholds.

## What's in `data/`

### `figure_<i>.csv`

Each file holds exactly the (already aggregated) rows the matching figure plots — bin centers,
PDF values, fitted parameters, per-period means — not individual users' raw trajectories.

* **figure_1 / figure_8** (`rg_and_dist`) — `panel` (`radius_of_gyration`|`jump_length`), `period`,
  `bin_center`, `pdf`, `fit_alpha`, `fit_sigma`, `fit_xmin`, `fit_fraction` (fit columns only
  populated for the `jump_length` panel — a power-law fit to the jump-length tail).
* **figure_2 / figure_9** (`rg_krg`) — `period`, `k`, `rg_km`, `krg_km`, `density`. Sparse: cells
  with zero density are omitted (the full grid is a fixed 40x40 bins over `[0.1, 1500]` km on both
  axes, reconstructed by the plot code).
* **figure_3 / figure_10** (`st_and_frequency`) — `panel` (`st`|`frequency`), `period`, `time_step`,
  `mean_st`, `std_st` (S(t) panel only), `rank`, `mean_frequency` (frequency panel only).
* **figure_4 / figure_11** (`gonzalez`) — `period`, `x_norm`, `y_norm`, `count`. Sparse: zero-count
  cells omitted (fixed 40x40 grid, `x_norm` in `[-1.5, 1.5]`, `y_norm` in `[-2.2, 2.2]`).
* **figure_5 / figure_12** (`sigmaxy`) — `period`, `axis` (`sigma_x`|`sigma_y`), `bin_center`, `pdf`,
  `mean_km`, `mean_pct_change_vs_baseline` (percent change vs. the pre-restriction period; `NaN` for
  the baseline period itself).
* **figure_6 / figure_13** (`entropy`) — `period`, `entropy_type`
  (`random_entropy`|`uncorrelated_entropy`|`real_entropy`|`scaled_entropy`), `bin_center`, `pdf`.
* **figure_7** (`weekly_rg`, California only) — `panel` (`rurality`|`party`), `week`, `stratum`
  (`rural`/`urban` or `Democratic`/`Republican`), `mean_rg`, `std_rg`.

### Sensitivity tables (`<feature>_t_threshold_{8,24}_MA-CA.csv`)

Figures 1–13 are only computed at `t_threshold = 1`; these seven tables show the same qualitative
story holds at coarser burst-filter thresholds (8 and 24 hours), for both regions in one file:

* `distance_powerlaw` — jump-length power-law fit (`alpha`, `alpha_sigma`, `xmin_km`,
  `tail_fraction`) per region/period.
* `entropy_peak` — PDF peak (mode) per region/period/entropy type.
* `frequency_top2_pct` — combined visitation share of the two most-visited locations, per
  region/period.
* `rg_distance_crossing` — the radius-of-gyration PDF crossing point (`r_g_star_km`) and area between
  curves for the restriction period vs. pre/post, per region.
* `rg_krg_bin0_percentile` — 95th percentile of `R_g` for users whose k-`R_g` falls in the
  near-zero bin, per region/period/k.
* `sigmaxy_pct_change` — mean sigma_x/sigma_y (km) and percent change vs. baseline, per
  region/period.
* `st_exponent_mu` — S(t) exploration-curve power-law exponent, per region/period.

## Methods

`metrics_change_mobility_covid.py` is a documented, from-scratch reference implementation of every
metric behind the figures and sensitivity tables above:

* **Radius of gyration** / **k-radius of gyration** — how far a user roams from their time-weighted
  centroid of activity; the k-variant restricts this to each user's `k` most-visited locations.
* **Entropy** (random / uncorrelated / real) — three increasingly informative measures of how
  predictable a user's sequence of visited locations is; real entropy uses the LZ78 estimator over
  the user's time-ordered location sequence.
* **Jump lengths / total distance** — the distribution of hop lengths between consecutive stops.
* **Gonzalez trajectory shape (PCA)** — rotates each user's stops into their own principal axes and
  normalizes by `sigma_x`/`sigma_y`, giving a shape- and scale-invariant 2-D density of where people
  go relative to their main axis of travel (figures 4/5/11/12). The module keeps both the current
  formula and the specific legacy formula that actually produced those figures (documented, bug-for-bug,
  in its docstring) for reproducibility.
* **S(t) exploration curve** — number of distinct locations a user has visited by hour `t`.
* **Location frequency by rank** — how concentrated a user's visits are on their most-visited places.
* **Weekly radius of gyration** — radius of gyration computed independently in rolling weekly
  windows, the time series behind figure_7.

### Important: raw mobility data is not included

These methods are written to operate on **raw, per-stop mobility data** — one row per stop, with a
user id, latitude/longitude, start/end time, and a geohash of the stop location (the exact schema is
documented at the top of `metrics_change_mobility_covid.py`). That raw data is **not distributed in
this repository**. What you have here is only the already-aggregated output needed to reproduce the
published figures and tables — `data/figure_<i>.csv` and the sensitivity tables — which is enough to
regenerate every plot, but not to rerun the metrics on individual trajectories.

If you do have compatible raw stop-point data, `metrics_change_mobility_covid.py` has no dependency
on anything else in this repository (only `numpy`/`pandas`/`polars`/`scipy`) — you can call its
functions directly against your own data.

## Where the data comes from

`extract_figure_data.py` documents, for reference, how `data/figure_<i>.csv` and the sensitivity
tables were originally derived from the raw mobility pipeline that produced this paper's results.
It is not runnable in this repository as-is (that pipeline and its raw data live elsewhere) and you
do not need it to reproduce any figure — the CSVs it would produce are already committed under
`data/`.
