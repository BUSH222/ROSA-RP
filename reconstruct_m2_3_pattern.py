#!/usr/bin/env python3
"""Reconstruct the Meteor-M N2-3 (M2-3) LRPT transmitting antenna pattern.

Pipeline
--------
1. Load and validate the per-pass CSVs for M2-3 and M2-4 (``dfs_filtered/m2_[34]_obs_*.csv``).
2. Rebuild each satellite's ECEF position from the ground-station az/alt/range (WGS84 + local ENU),
   then linearly interpolate it to the exact time of every 0.1 s power sample.
3. Stage 1 - receiver calibration: for every M2-4 sample, remove the known M2-4 transmit gain and the
   free-space term to get R = C + G_r (receiver directional response plus an unknown common constant).
   Median-bin R on a 1 deg ground-station (azimuth, elevation) grid and build a gap-limited linear
   interpolator over the populated bins.
4. Stage 2 - for every M2-3 sample, evaluate that map at the sample's ground-station direction and
   recover G_t3 = p3 - L(d3) - R_hat - (P_t3 - P_t4). The constant C cancels.
5. Stage 3 - express each M2-3 sample in an idealised satellite body frame (z = nadir, x = direction of
   travel), median-bin the recovered gains on a 1 deg (azimuth, off-nadir) grid, and fill empty grid
   nodes by gap-limited linear interpolation.
6. Write the grid CSV, polar heatmaps, a receiver-map product, metadata JSON and a diagnostics summary.

Measurement model (dB)
----------------------
    p_ij = C + G_t,ij + G_r,j + L(d_ij),     L(d) = 20 log10( lambda / (4 pi d) )

``p`` is the supplied ``power_db`` in ARBITRARY units. ``C`` absorbs the unknown receiver-chain calibration and
cancels between M2-4 and M2-3, so no absolute received-power calibration is needed: this is a gain-comparison
(gain-transfer) measurement with M2-4 as the reference antenna,

    G_t3 = G_t4 + (p3 - p4) + (L(d4) - L(d3)) - (P_t3 - P_t4)     at the same ground-station direction.

The recovered M2-3 gain therefore inherits the absolute scale of the supplied M2-4 pattern (``Ku_dB``) and is
reported in GAIN_UNIT (dBic by default). Its accuracy is limited by the four uncertainty sources listed in the
output metadata (``uncertainty_sources``), not by the unknown C.

Coordinate conventions
----------------------
Ground-station frame (used for geometry and the receiver calibration map):
    az_deg  - azimuth, clockwise from geodetic north, [0, 360).
    alt_deg - elevation above the local (WGS84 ellipsoid-normal) horizon.
    ENU line of sight: u = [cos(alt) sin(az), cos(alt) cos(az), sin(alt)].

Satellite body frame (used for the M2-3 pattern; an IDEALISED geometric frame, not measured attitude):
    z_b = n = -r_sat / |r_sat|             geocentric nadir (toward Earth's centre).
    x_b = f = unit(v - (v.n) n)            direction of travel, projected perpendicular to nadir.
    y_b = z_b x x_b                        right-handed; with z down and x forward, +y_b is starboard.
    q   = unit(r_GS - r_sat)               satellite -> ground-station direction.
    azimuth_deg   = atan2(q.y_b, q.x_b) wrapped to [0, 360); 0 = forward, 90 = +y_b.
    off_nadir_deg = atan2(sqrt(qx^2 + qy^2), q.z_b); 0 = nadir.

Note: the task text writes the nadir as unit(r_GS - r_sat), which is identical to q and would make every
off-nadir angle zero. Per the stated intent ("satellite-to-Earth-centre/nadir direction"), and as agreed
with the user, the geocentric nadir -r_sat/|r_sat| is used.

Run
---
    python reconstruct_m2_3_pattern.py [--data-dir dfs_filtered] [--pattern m2_4_pattern.csv]
                                       [--output-dir m2_3_pattern_output] [--write-samples]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.spatial import Delaunay

# =====================================================================================================
# CONFIGURATION - every constant, threshold and convention used by the pipeline lives here.
# =====================================================================================================

# --- Paths --------------------------------------------------------------------------------------------
DATA_DIR = Path("./dfs_filtered")
M2_4_PATTERN_PATH = Path("./m2_4_pattern.csv")
OUTPUT_DIR = Path("./m2_3_pattern_output")
PATTERN_CSV_NAME = "m2_3_antenna_pattern.csv"

# --- Gain scale ---------------------------------------------------------------------------------------------
# The M2-3 gain inherits the unit of the supplied M2-4 reference (Ku_dB). Both links are RHCP, so the co-polar
# gain is labelled dBic (numerically equal to dBi for a perfectly circularly polarised antenna).
# Set to "dBi" if the reference pattern is quoted in dBi. Used for column names, labels and metadata.
GAIN_UNIT = "dBic"
GAIN_COLUMN = f"gain_{GAIN_UNIT.lower()}"

# --- Physical constants -------------------------------------------------------------------------------
# Transmit power delivered to each satellite's antenna (W). Nominally equal; any difference shifts the whole M2-3
# pattern dB-for-dB (uncertainty source 1) and is corrected as G_t3 -= 10 log10(PT_M2_3_W / PT_M2_4_W).
PT_M2_3_W = 6.5
PT_M2_4_W = 6.5
PT_DBW = 10.0 * np.log10(PT_M2_4_W)  # ~8.13 dBW
TX_POWER_CORRECTION_DB = 10.0 * np.log10(PT_M2_3_W / PT_M2_4_W)  # 0 dB while the powers are equal
FREQ_HZ = 137.9e6  # LRPT carrier frequency
WAVELENGTH_M = 2.173984  # c / f as given in the task

# Ground station (WGS84 geodetic). The 127 m "above sea level" is used as ellipsoidal height, matching the
# Skyfield ``wgs84.latlon(elevation_m=127)`` call that generated az/alt/range (geoid offset ~15 m, negligible).
GS_LAT_DEG = 55.5269
GS_LON_DEG = 37.0888
GS_ALT_M = 127.0

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
OMEGA_EARTH_RAD_S = 7.2921150e-5  # only used when HEADING_VELOCITY_FRAME == "inertial"

# --- Input validation -----------------------------------------------------------------------------------
NUMERIC_COLUMNS = [
    "unix_time",
    "az_deg",
    "alt_deg",
    "range_km",
    "time_s",
    "p_total",
    "p_noise_scaled",
    "power_linear",
    "power_db",
]
REQUIRED_COLUMNS = {"time", "observation", *NUMERIC_COLUMNS}

# power_linear must equal p_total - p_noise_scaled within max(ATOL, RTOL * |power_linear|).
# The CSVs store ~7 significant digits; observed residuals are <= 4e-8 absolute / 1e-4 relative.
POWER_LINEAR_ATOL = 1e-7
POWER_LINEAR_RTOL = 1e-3
# power_db must equal 10 log10(power_linear) within this many dB (observed residuals <= 2.2e-4 dB).
POWER_DB_ATOL = 1e-3

# Valid slant range (km). Both satellites orbit at ~830 km, so the horizon slant range is ~3.4e3 km.
RANGE_KM_MIN_EXCLUSIVE = 0.0
RANGE_KM_MAX = 5000.0

# Angles outside az in [0, 360] or alt in [-90, 90] are "invalid_angle". Separately, rows below this
# elevation mask are rejected as "below_elevation_mask" (they are AOS/LOS ephemeris artefacts at ~-0.002 deg).
MIN_ELEVATION_DEG = 0.0

# Precedence of rejection reasons: each rejected row is counted once, under the first reason that applies.
REJECTION_REASONS = [
    "non_finite_value",
    "invalid_range",
    "invalid_angle",
    "below_elevation_mask",
    "non_positive_power",
    "power_linear_mismatch",
    "power_db_mismatch",
]

# --- Geometry / timing ------------------------------------------------------------------------------------
# az/alt/range are sampled once per second (at AOS + k s) and were merged onto the 0.1 s power samples by
# floor(time_s). Each sample's true time is AOS + time_s = unix_time + frac(time_s). When enabled, the ECEF
# position (and travel direction) is linearly interpolated between the surrounding per-second positions of
# the same track. Interpolation never bridges a gap larger than GEOMETRY_INTERP_MAX_GAP_S; where it cannot
# be done (track end, large gap) the supplied per-second geometry is used and the sample is counted.
INTERPOLATE_GEOMETRY_TO_SAMPLE_TIME = True
GEOMETRY_INTERP_MAX_GAP_S = 10.0
SAMPLE_TIME_OFFSET_TOL_S = 1e-3  # (unix_time - floor(time_s)) must be constant within a track
TIME_S_FLOOR_EPS = 1e-6  # guards floor(time_s) against values like 2.9999999

# Direction of travel. Central chord r[i+1] - r[i-1] when both neighbours are within the gap limit,
# one-sided displacement otherwise, nearest valid heading of the same track if neither neighbour is usable.
HEADING_MAX_GAP_S = 10.0
# "ecef": velocity relative to the rotating Earth, from reconstructed ECEF positions (as specified).
# "inertial": adds omega_E x r (orbital-plane velocity). The two differ by ~2-3.5 deg in heading here.
HEADING_VELOCITY_FRAME = "ecef"
HEADING_DEGENERATE_REL_TOL = 1e-6  # reject if |v_perp| < tol * |v|
ORTHONORMAL_TOL = 1e-9  # body-frame orthonormality / handedness check

# --- Interpolation / aggregation ------------------------------------------------------------------------
# Both stages use the same method (see GapLimitedLinearInterpolator):
#   * samples are assigned to the nearest node of a 1 deg grid and aggregated by the median;
#   * populated nodes are the support points of a Delaunay triangulation in (azimuth, polar-angle) degrees,
#     with copies of nodes within AZ_WRAP_PAD_DEG of 0/360 deg appended at +/-360 deg (circular azimuth);
#   * a query point gets the barycentric (linear) value of its triangle only if the support points that
#     actually carry weight (barycentric weight > BARYCENTRIC_EPS) span <= MAX_INTERP_GAP_DEG in azimuth
#     AND <= MAX_INTERP_GAP_DEG in the polar angle. Outside the triangulation -> NaN (no extrapolation).
MAX_INTERP_GAP_DEG = 5.0
GAP_TOL_DEG = 1e-9
BARYCENTRIC_EPS = 1e-9

RX_MAP_NODE_STEP_DEG = 1.0  # ground-station-frame calibration-map node spacing (az and elevation)
RX_MAP_MIN_SAMPLES = 1  # minimum M2-4 samples for a calibration node to be used as support
PATTERN_MIN_SAMPLES = 1  # minimum M2-3 samples for an output node to be "measured"
AZ_WRAP_PAD_DEG = MAX_INTERP_GAP_DEG + 1.0

# --- Output grid ------------------------------------------------------------------------------------------
GRID_AZIMUTH_DEG = np.arange(0, 360, 1)  # 0..359 inclusive
GRID_OFF_NADIR_DEG = np.arange(0, 63, 1)  # 0..62 inclusive

# --- Extra validation ---------------------------------------------------------------------------------------
# Hold-out closure test: calibrate on M2-4 tracks with even observation numbers, then recover the M2-4
# pattern from the odd-numbered tracks and compare with the known reference.
RUN_CLOSURE_TEST = True
CLOSURE_QUADRANT_MIN_OFF_NADIR_DEG = 20.0  # quadrant breakdown only where body azimuth is well defined
UNCERTAINTY_NOTEBOOK = "find_m24_pattern_uncertainty.ipynb"  # quantifies uncertainty source 4 per output cell

# --- Plotting -----------------------------------------------------------------------------------------------
GAIN_CMAP = ListedColormap(plt.get_cmap("Reds")(np.linspace(0.12, 1.0, 256)))  # higher gain = redder
RX_CMAP = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.12, 1.0, 256)))
MISSING_COLOR = "#c3c2b7"  # neutral gray for unsupported (NaN) cells
COLOR_PERCENTILES = (1.0, 99.0)  # robust colour limits; the colour bar is extended beyond them
PLOT_DPI = 200

SATELLITES = {"m2_3": "Meteor-M N2-3", "m2_4": "Meteor-M N2-4"}


# =====================================================================================================
# Loading and validation
# =====================================================================================================


def obs_number(path: Path) -> int:
    match = re.search(r"_obs_(\d+)\.csv$", path.name)
    return int(match.group(1)) if match else -1


def load_observations(data_dir: Path, sat: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all ``{sat}_obs_*.csv`` files. Returns (rows, per-file summary). Each file is one track."""
    paths = sorted(data_dir.glob(f"{sat}_obs_*.csv"), key=obs_number)
    if not paths:
        raise FileNotFoundError(f"No {sat}_obs_*.csv files in {data_dir.resolve()}")

    frames, summary = [], []
    for path in paths:
        try:
            df = pd.read_csv(path, dtype={"time": str, "observation": str})
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        n_obs_ids = df["observation"].nunique()
        if n_obs_ids > 1:
            raise ValueError(f"{path.name}: contains {n_obs_ids} observation ids; expected one track per file")
        summary.append(
            {
                "satellite": sat,
                "file": path.name,
                "obs_number": obs_number(path),
                "observation": df["observation"].iloc[0] if len(df) else "",
                "n_rows": len(df),
                "first_time": df["time"].iloc[0] if len(df) else "",
                "last_time": df["time"].iloc[-1] if len(df) else "",
            }
        )
        if len(df):
            df = df.copy()
            df["file"] = path.name
            df["obs_number"] = obs_number(path)
            frames.append(df)

    rows = pd.concat(frames, ignore_index=True)
    for col in NUMERIC_COLUMNS:
        rows[col] = pd.to_numeric(rows[col], errors="coerce").astype(float)
    rows["satellite"] = sat
    return rows, pd.DataFrame(summary)


def validate_rows(df: pd.DataFrame) -> pd.Series:
    """Return a per-row rejection reason ('' = accepted). Fully vectorised; one reason per row (precedence)."""
    values = df[NUMERIC_COLUMNS].to_numpy(dtype=float)
    az, alt, rng = df["az_deg"].to_numpy(), df["alt_deg"].to_numpy(), df["range_km"].to_numpy()
    p_lin, p_db = df["power_linear"].to_numpy(), df["power_db"].to_numpy()
    p_diff = df["p_total"].to_numpy() - df["p_noise_scaled"].to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        db_from_lin = 10.0 * np.log10(np.where(p_lin > 0, p_lin, np.nan))
        checks = {
            "non_finite_value": ~np.isfinite(values).all(axis=1),
            "invalid_range": ~((rng > RANGE_KM_MIN_EXCLUSIVE) & (rng <= RANGE_KM_MAX)),
            "invalid_angle": ~((az >= 0) & (az <= 360) & (alt >= -90) & (alt <= 90)),
            "below_elevation_mask": alt < MIN_ELEVATION_DEG,
            # Non-positive power is rejected, never replaced by a small positive number.
            "non_positive_power": ~(p_lin > 0),
            "power_linear_mismatch": np.abs(p_lin - p_diff)
            > np.maximum(POWER_LINEAR_ATOL, POWER_LINEAR_RTOL * np.abs(p_lin)),
            "power_db_mismatch": np.abs(p_db - db_from_lin) > POWER_DB_ATOL,
        }

    reason = np.full(len(df), "", dtype=object)
    for name in REJECTION_REASONS:
        reason[(reason == "") & checks[name]] = name
    return pd.Series(reason, index=df.index, name="reject_reason")


def rejection_tables(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rejected = rows[rows["reject_reason"] != ""]
    by_reason = {r: int((rejected["reject_reason"] == r).sum()) for r in REJECTION_REASONS}
    by_file = (
        rejected.groupby(["satellite", "file", "observation", "reject_reason"], observed=True)
        .size()
        .rename("n_rows")
        .reset_index()
    )
    return by_reason, by_file


# =====================================================================================================
# Coordinate transformations (all lengths in metres, angles in degrees unless noted)
# =====================================================================================================


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    n = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + h_m) * np.cos(lat) * np.cos(lon),
            (n + h_m) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - WGS84_E2) + h_m) * np.sin(lat),
        ]
    )


def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rows are the local East, North, Up unit vectors expressed in ECEF."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def azalt_range_to_ecef(az_deg, alt_deg, range_m, r_gs: np.ndarray, enu: np.ndarray) -> np.ndarray:
    """r_sat = r_GS + d * u_ECEF, with u_ENU = [cos(alt) sin(az), cos(alt) cos(az), sin(alt)]."""
    az, alt = np.radians(az_deg), np.radians(alt_deg)
    u_enu = np.column_stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)])
    u_ecef = u_enu @ enu  # enu rows are E, N, U in ECEF -> this is e*E + n*N + u*U
    return r_gs + np.asarray(range_m)[:, None] * u_ecef


def ecef_to_azalt_range(r_sat: np.ndarray, r_gs: np.ndarray, enu: np.ndarray):
    d = r_sat - r_gs
    e, n, u = (d @ enu.T).T
    rng = np.linalg.norm(d, axis=1)
    az = np.mod(np.degrees(np.arctan2(e, n)), 360.0)
    alt = np.degrees(np.arctan2(u, np.hypot(e, n)))
    return az, alt, rng


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def geocentric_nadir(r_sat: np.ndarray) -> np.ndarray:
    return -unit(r_sat)


def off_nadir_deg(r_sat: np.ndarray, r_gs: np.ndarray) -> np.ndarray:
    """Angle between satellite->GS and geocentric nadir (numerically stable atan2 form)."""
    q = unit(r_gs - r_sat)
    n = geocentric_nadir(r_sat)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(q, n), axis=1), np.einsum("ij,ij->i", q, n)))


def free_space_term_db(d_m: np.ndarray) -> np.ndarray:
    """L(d) = 20 log10(lambda / (4 pi d)), d in metres (negative number, ~-123 dB at 1000 km)."""
    return 20.0 * np.log10(WAVELENGTH_M / (4.0 * np.pi * d_m))


# =====================================================================================================
# Track geometry: per-second knots, heading estimation, interpolation to sample time
# =====================================================================================================


@dataclass
class Knots:
    """Unique per-second positions of every track, sorted by (file, unix_time)."""

    file: np.ndarray
    t: np.ndarray
    r: np.ndarray  # (n, 3) ECEF metres
    heading: np.ndarray | None = None  # (n, 3) unit vectors, NaN when invalid
    heading_status: np.ndarray | None = None  # central / forward / backward / borrowed / invalid
    borrow_dt_s: np.ndarray | None = None


def build_knots(df: pd.DataFrame, r_gs: np.ndarray, enu: np.ndarray) -> tuple[Knots, np.ndarray, int]:
    """Collapse rows that share (file, unix_time) into one position knot.

    Returns the knots, each row's knot index, and the number of (file, unix_time) groups whose rows
    disagree on geometry (should be 0; the first row's geometry is used if not).
    """
    geo = df[["file", "unix_time", "az_deg", "alt_deg", "range_km"]]
    grouped = geo.groupby(["file", "unix_time"], sort=True)
    first = grouped.first()
    spread = grouped.max() - grouped.min()
    n_conflicts = int((spread > 1e-9).any(axis=1).sum())

    r = azalt_range_to_ecef(
        first["az_deg"].to_numpy(), first["alt_deg"].to_numpy(), first["range_km"].to_numpy() * 1000.0, r_gs, enu
    )
    knots = Knots(
        file=first.index.get_level_values("file").to_numpy(),
        t=first.index.get_level_values("unix_time").to_numpy(dtype=float),
        r=r,
    )
    key = pd.MultiIndex.from_frame(df[["file", "unix_time"]])
    row_knot = first.index.get_indexer(key)
    assert (row_knot >= 0).all()
    return knots, row_knot, n_conflicts


def estimate_knot_headings(knots: Knots) -> None:
    """Direction of travel at every knot (unit ECEF vector, radial component kept here).

    Central chord r[i+1] - r[i-1] if both neighbours are in the same track and each step is <= HEADING_MAX_GAP_S;
    one-sided displacement if only one neighbour qualifies; otherwise borrow the time-nearest valid heading
    of the same track (recorded); if none exists the knot is invalid.
    """
    n = len(knots.t)
    same_prev = np.r_[False, knots.file[1:] == knots.file[:-1]]
    same_next = np.r_[knots.file[:-1] == knots.file[1:], False]
    dt_prev = np.full(n, np.inf)
    dt_next = np.full(n, np.inf)
    dt_prev[1:] = np.where(same_prev[1:], knots.t[1:] - knots.t[:-1], np.inf)
    dt_next[:-1] = np.where(same_next[:-1], knots.t[1:] - knots.t[:-1], np.inf)
    ok_prev, ok_next = dt_prev <= HEADING_MAX_GAP_S, dt_next <= HEADING_MAX_GAP_S

    r = knots.r
    r_prev = np.vstack([r[:1], r[:-1]])
    r_next = np.vstack([r[1:], r[-1:]])
    v = np.full((n, 3), np.nan)
    status = np.full(n, "invalid", dtype=object)

    central = ok_prev & ok_next
    forward = ~ok_prev & ok_next
    backward = ok_prev & ~ok_next
    v[central] = (r_next[central] - r_prev[central]) / (dt_prev[central] + dt_next[central])[:, None]
    v[forward] = (r_next[forward] - r[forward]) / dt_next[forward][:, None]
    v[backward] = (r[backward] - r_prev[backward]) / dt_prev[backward][:, None]
    status[central], status[forward], status[backward] = "central", "forward", "backward"

    if HEADING_VELOCITY_FRAME == "inertial":
        v = v + np.cross(np.array([0.0, 0.0, OMEGA_EARTH_RAD_S]), r)
    elif HEADING_VELOCITY_FRAME != "ecef":
        raise ValueError(f"Unknown HEADING_VELOCITY_FRAME {HEADING_VELOCITY_FRAME!r}")

    # Reject numerically degenerate headings (velocity almost parallel to nadir) at the knot itself.
    nadir = geocentric_nadir(r)
    v_perp = v - np.einsum("ij,ij->i", v, nadir)[:, None] * nadir
    with np.errstate(invalid="ignore"):
        degenerate = np.linalg.norm(v_perp, axis=1) < HEADING_DEGENERATE_REL_TOL * np.linalg.norm(v, axis=1)
    has_direct = np.isin(status, ["central", "forward", "backward"]) & ~degenerate
    status[~has_direct] = "invalid"
    v[~has_direct] = np.nan

    # Borrow the time-nearest valid heading from the same track.
    borrow_dt = np.full(n, np.nan)
    for i in np.flatnonzero(~has_direct):
        candidates = np.flatnonzero((knots.file == knots.file[i]) & has_direct)
        if candidates.size == 0:
            continue
        j = candidates[np.argmin(np.abs(knots.t[candidates] - knots.t[i]))]
        v[i] = v[j]
        status[i] = "borrowed"
        borrow_dt[i] = abs(knots.t[j] - knots.t[i])

    knots.heading = unit(v)
    knots.heading_status = status
    knots.borrow_dt_s = borrow_dt


def sample_time_offsets(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Per-row measurement time AOS + time_s, where AOS = unix_time - floor(time_s) (constant per track).

    Returns the measurement times (NaN for tracks whose offset is not constant) and the offending files.
    """
    offset = df["unix_time"].to_numpy() - np.floor(df["time_s"].to_numpy() + TIME_S_FLOOR_EPS)
    by_file = pd.Series(offset, index=df.index).groupby(df["file"])
    spread = by_file.max() - by_file.min()
    bad_files = spread[spread > SAMPLE_TIME_OFFSET_TOL_S].index.tolist()
    aos = by_file.transform("median").to_numpy()
    t_meas = aos + df["time_s"].to_numpy()
    t_meas[df["file"].isin(bad_files).to_numpy()] = np.nan
    return t_meas, bad_files


def sample_geometry(df: pd.DataFrame, knots: Knots, row_knot: np.ndarray) -> dict:
    """ECEF position (and heading if available) for every row, interpolated to the sample time if enabled.

    Interpolation uses the row's own knot k (t_k = its unix_time) and the next knot k+1 of the same track,
    provided t_{k+1} - t_k <= GEOMETRY_INTERP_MAX_GAP_S. Otherwise the knot-k (supplied) geometry is used.
    """
    n_rows = len(df)
    k = row_knot
    r = knots.r[k].copy()
    has_heading = knots.heading is not None
    heading = knots.heading[k].copy() if has_heading else None
    heading_status = knots.heading_status[k].copy() if has_heading else None
    geom_status = np.full(n_rows, "supplied", dtype=object)
    bad_offset_files: list[str] = []

    if INTERPOLATE_GEOMETRY_TO_SAMPLE_TIME:
        t_meas, bad_offset_files = sample_time_offsets(df)
        frac = t_meas - knots.t[k]
        exact = np.abs(frac) < 1e-9
        kn = np.minimum(k + 1, len(knots.t) - 1)
        same_track = (k + 1 < len(knots.t)) & (knots.file[kn] == knots.file[k])
        gap = knots.t[kn] - knots.t[k]
        can = ~exact & np.isfinite(frac) & (frac > 0) & same_track & (gap <= GEOMETRY_INTERP_MAX_GAP_S)
        can &= frac < gap  # sample must lie strictly inside the bracket
        w = np.where(can, frac / np.where(gap > 0, gap, 1.0), 0.0)[:, None]
        r = np.where(can[:, None], (1.0 - w) * knots.r[k] + w * knots.r[kn], r)
        if has_heading:
            h_next = knots.heading[kn]
            blended = unit((1.0 - w) * knots.heading[k] + w * h_next)
            heading = np.where(can[:, None], blended, heading)
            # Status of an interpolated heading: invalid if either knot is invalid, borrowed if either is borrowed.
            st_next = knots.heading_status[kn]
            either_invalid = (heading_status == "invalid") | (st_next == "invalid")
            either_borrowed = (heading_status == "borrowed") | (st_next == "borrowed")
            heading_status = np.where(
                can & either_invalid, "invalid", np.where(can & either_borrowed, "borrowed", heading_status)
            )
        geom_status[exact] = "exact_knot"
        geom_status[can] = "interpolated"
        geom_status[~exact & ~can] = "fallback_supplied"

    return {
        "r_sat": r,
        "heading": heading,
        "heading_status": heading_status,
        "geometry_status": geom_status,
        "bad_offset_files": bad_offset_files,
    }


def body_frame_angles(r_sat: np.ndarray, r_gs: np.ndarray, heading: np.ndarray) -> dict:
    """Body-frame azimuth / off-nadir of the ground station, plus validity and orthonormality diagnostics."""
    z_b = geocentric_nadir(r_sat)
    v_perp = heading - np.einsum("ij,ij->i", heading, z_b)[:, None] * z_b
    norm_perp = np.linalg.norm(v_perp, axis=1)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(norm_perp) & (norm_perp >= HEADING_DEGENERATE_REL_TOL)
    x_b = np.full_like(v_perp, np.nan)
    x_b[valid] = v_perp[valid] / norm_perp[valid, None]
    y_b = unit(np.cross(z_b, x_b))

    q = unit(r_gs - r_sat)
    qx, qy, qz = (np.einsum("ij,ij->i", q, a) for a in (x_b, y_b, z_b))
    az = np.mod(np.degrees(np.arctan2(qy, qx)), 360.0)
    th = np.degrees(np.arctan2(np.hypot(qx, qy), qz))

    # Orthonormality and right-handedness: x.y, y.z, z.x ~ 0, |axes| ~ 1, det[x y z] ~ +1.
    v = valid
    dots = np.abs(
        np.stack(
            [
                np.einsum("ij,ij->i", x_b[v], y_b[v]),
                np.einsum("ij,ij->i", y_b[v], z_b[v]),
                np.einsum("ij,ij->i", z_b[v], x_b[v]),
            ]
        )
    )
    norms = np.abs(np.stack([np.linalg.norm(a[v], axis=1) - 1.0 for a in (x_b, y_b, z_b)]))
    det = np.einsum("ij,ij->i", np.cross(x_b[v], y_b[v]), z_b[v])
    ortho = {
        "max_abs_axis_dot": float(dots.max()) if dots.size else float("nan"),
        "max_abs_norm_error": float(norms.max()) if norms.size else float("nan"),
        "max_abs_det_minus_1": float(np.abs(det - 1.0).max()) if det.size else float("nan"),
    }
    if any(val > ORTHONORMAL_TOL for val in ortho.values()):
        raise RuntimeError(f"Body frame failed the orthonormality check: {ortho}")
    az[~valid], th[~valid] = np.nan, np.nan
    return {"azimuth_deg": az, "off_nadir_deg": th, "valid": valid, "orthonormality": ortho}


# =====================================================================================================
# Reference pattern
# =====================================================================================================


@dataclass
class ReferencePattern:
    theta_deg: np.ndarray  # sorted, with a 0 deg sample prepended
    gain_db: np.ndarray
    first_sample_deg: float
    max_theta_deg: float

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """Linear interpolation; NaN outside [0, max supplied angle] (no extrapolation)."""
        theta = np.asarray(theta, dtype=float)
        out = np.interp(theta, self.theta_deg, self.gain_db)
        out[~((theta >= 0.0) & (theta <= self.max_theta_deg))] = np.nan
        return out


def load_reference_pattern(path: Path) -> ReferencePattern:
    ref = pd.read_csv(path)
    if not {"theta_deg", "Ku_dB"} <= set(ref.columns):
        raise ValueError(f"{path}: expected columns theta_deg, Ku_dB")
    ref = ref[["theta_deg", "Ku_dB"]].astype(float).dropna().sort_values("theta_deg")
    if ref["theta_deg"].duplicated().any():
        raise ValueError(f"{path}: duplicate theta_deg values")
    th, g = ref["theta_deg"].to_numpy(), ref["Ku_dB"].to_numpy()
    # The first sample (~1.03 deg) is used as the boresight (0 deg) gain.
    return ReferencePattern(np.r_[0.0, th], np.r_[g[0], g], float(th[0]), float(th[-1]))


# =====================================================================================================
# Aggregation and gap-limited interpolation
# =====================================================================================================


def bin_median(az_deg, polar_deg, values, polar_nodes: np.ndarray, step: float = 1.0) -> pd.DataFrame:
    """Median and count per nearest grid node. Azimuth nodes 0..360-step wrap; polar nodes = ``polar_nodes``.

    Nearest node = round(angle / step) * step (numpy round-half-to-even). Samples whose polar node falls
    outside ``polar_nodes`` are dropped (and counted by the caller).
    """
    n_az = round(360.0 / step)
    az_idx = np.mod(np.rint(np.asarray(az_deg) / step).astype(int), n_az)
    pol_idx = np.rint((np.asarray(polar_deg) - polar_nodes[0]) / step).astype(int)
    inside = (pol_idx >= 0) & (pol_idx < len(polar_nodes))
    cell = pol_idx[inside] * n_az + az_idx[inside]
    s = pd.Series(np.asarray(values)[inside]).groupby(cell)
    out = pd.DataFrame({"median": s.median(), "count": s.size()})
    out["az_node"] = (out.index.to_numpy() % n_az) * step
    out["polar_node"] = polar_nodes[out.index.to_numpy() // n_az]
    return out.reset_index(drop=True)


class GapLimitedLinearInterpolator:
    """Linear (barycentric) interpolation over scattered support points in (azimuth, polar) degrees.

    * Circular azimuth: support points within ``az_pad`` of 0/360 deg are duplicated at +/-360 deg before the
      Delaunay triangulation, so triangles span the 0/360 seam; queries are wrapped to [0, 360).
    * No extrapolation: queries outside the triangulation's convex hull return NaN (status 1).
    * Gap limit: for the triangle containing the query, only vertices with barycentric weight > eps actually
      contribute. The query is accepted only if those contributing vertices span <= ``max_gap`` deg in azimuth
      AND <= ``max_gap`` deg in the polar angle; otherwise NaN (status 2). A query that coincides with a support
      point therefore always returns that point's own value.
    """

    OK, OUTSIDE_HULL, GAP_EXCEEDED = 0, 1, 2

    def __init__(self, az_deg, polar_deg, values, max_gap: float, az_pad: float):
        az = np.mod(np.asarray(az_deg, dtype=float), 360.0)
        pol = np.asarray(polar_deg, dtype=float)
        val = np.asarray(values, dtype=float)
        lo, hi = az < az_pad, az >= 360.0 - az_pad
        self.points = np.column_stack([np.r_[az, az[lo] + 360.0, az[hi] - 360.0], np.r_[pol, pol[lo], pol[hi]]])
        self.values = np.r_[val, val[lo], val[hi]]
        self.max_gap = max_gap
        self.tri = Delaunay(self.points)

    def __call__(self, az_deg, polar_deg) -> tuple[np.ndarray, np.ndarray]:
        q = np.column_stack([np.mod(np.asarray(az_deg, dtype=float), 360.0), np.asarray(polar_deg, dtype=float)])
        out = np.full(len(q), np.nan)
        status = np.full(len(q), self.OUTSIDE_HULL, dtype=np.int8)
        finite = np.isfinite(q).all(axis=1)
        simplex = np.full(len(q), -1)
        simplex[finite] = self.tri.find_simplex(q[finite])
        inside = simplex >= 0
        if not inside.any():
            return out, status

        s = simplex[inside]
        transform = self.tri.transform[s]
        b = np.einsum("nij,nj->ni", transform[:, :2, :], q[inside] - transform[:, 2, :])
        bary = np.clip(np.column_stack([b, 1.0 - b.sum(axis=1)]), 0.0, None)
        bary /= bary.sum(axis=1, keepdims=True)
        verts = self.tri.simplices[s]
        contributing = bary > BARYCENTRIC_EPS

        def extent(coord: np.ndarray) -> np.ndarray:
            hi_ = np.where(contributing, coord, -np.inf).max(axis=1)
            lo_ = np.where(contributing, coord, np.inf).min(axis=1)
            return hi_ - lo_

        az_ext = extent(self.points[verts, 0])
        pol_ext = extent(self.points[verts, 1])
        ok = (az_ext <= self.max_gap + GAP_TOL_DEG) & (pol_ext <= self.max_gap + GAP_TOL_DEG)
        vals = (bary * self.values[verts]).sum(axis=1)

        idx = np.flatnonzero(inside)
        out[idx[ok]] = vals[ok]
        status[idx[ok]] = self.OK
        status[idx[~ok]] = self.GAP_EXCEEDED
        return out, status


def interpolation_method_text(stage: str, node_text: str) -> str:
    return (
        f"{stage}: samples median-aggregated to the nearest node of a {node_text}; populated nodes are support "
        "points of a scipy.spatial.Delaunay triangulation in (azimuth, polar) degrees with azimuth copies at "
        f"+/-360 deg for nodes within {AZ_WRAP_PAD_DEG:g} deg of the seam; barycentric linear interpolation; "
        f"a query is accepted only if the vertices with non-zero barycentric weight span <= {MAX_INTERP_GAP_DEG:g} "
        f"deg in azimuth and <= {MAX_INTERP_GAP_DEG:g} deg in the polar angle; outside the convex hull -> NaN "
        "(no extrapolation)."
    )


# =====================================================================================================
# Stage 1: receiver calibration from M2-4
# =====================================================================================================


def receiver_response_samples(m24: pd.DataFrame, ref: ReferencePattern) -> pd.DataFrame:
    """R_j = p_4j - G_t4(theta_j) - L(d_j) for each M2-4 sample (theta_j = off-boresight = off-nadir)."""
    out = m24.copy()
    out["ref_gain_db"] = ref(out["off_boresight_deg"].to_numpy())
    out["fspl_term_db"] = free_space_term_db(out["range_m"].to_numpy())
    out["rx_response_db"] = out["power_db"] - out["ref_gain_db"] - out["fspl_term_db"]
    return out


@dataclass
class ReceiverMap:
    interpolator: GapLimitedLinearInterpolator
    nodes: pd.DataFrame  # populated nodes used as support: az_node, polar_node (= elevation), median, count


def build_receiver_map(samples: pd.DataFrame) -> ReceiverMap:
    """Median-bin R on a ground-station (azimuth, elevation) grid and build the gap-limited interpolator."""
    el_nodes = np.arange(0.0, 90.0 + RX_MAP_NODE_STEP_DEG / 2, RX_MAP_NODE_STEP_DEG)
    nodes = bin_median(
        samples["gs_az_deg"], samples["gs_alt_deg"], samples["rx_response_db"], el_nodes, RX_MAP_NODE_STEP_DEG
    )
    nodes = nodes[nodes["count"] >= RX_MAP_MIN_SAMPLES].reset_index(drop=True)
    interp = GapLimitedLinearInterpolator(
        nodes["az_node"], nodes["polar_node"], nodes["median"], MAX_INTERP_GAP_DEG, AZ_WRAP_PAD_DEG
    )
    return ReceiverMap(interp, nodes)


def receiver_map_grid(rx: ReceiverMap) -> pd.DataFrame:
    """Evaluate the calibration map on its own regular node grid (for coverage reporting and plotting)."""
    az = np.arange(0.0, 360.0, RX_MAP_NODE_STEP_DEG)
    el = np.arange(0.0, 90.0 + RX_MAP_NODE_STEP_DEG / 2, RX_MAP_NODE_STEP_DEG)
    az_m, el_m = np.meshgrid(az, el)
    val, status = rx.interpolator(az_m.ravel(), el_m.ravel())
    grid = pd.DataFrame({"gs_az_deg": az_m.ravel(), "gs_alt_deg": el_m.ravel(), "rx_response_db": val})
    counts = rx.nodes.set_index(["az_node", "polar_node"])["count"]
    key = pd.MultiIndex.from_arrays([grid["gs_az_deg"], grid["gs_alt_deg"]])
    grid["sample_count"] = counts.reindex(key).fillna(0).astype(int).to_numpy()
    grid["status"] = np.select(
        [
            grid["sample_count"] > 0,
            status == GapLimitedLinearInterpolator.OK,
            status == GapLimitedLinearInterpolator.GAP_EXCEEDED,
        ],
        ["measured", "interpolated", "unsupported_gap"],
        "unsupported_outside_hull",
    )
    return grid


# =====================================================================================================
# Stage 2 and 3: M2-3 gain reconstruction and body-frame pattern
# =====================================================================================================


def reconstruct_m23_gain(m23: pd.DataFrame, rx: ReceiverMap) -> pd.DataFrame:
    """G_t3 = p_3 - L(d_3) - R_hat(az, alt) - (P_t3 - P_t4), R_hat from the M2-4 map (NaN if unsupported).

    The result is on the absolute scale of the M2-4 reference pattern (GAIN_UNIT); stored as ``gain_db``.
    """
    out = m23.copy()
    r_hat, status = rx.interpolator(out["gs_az_deg"].to_numpy(), out["gs_alt_deg"].to_numpy())
    out["rx_response_hat_db"] = r_hat
    out["rx_map_status"] = np.select(
        [status == GapLimitedLinearInterpolator.OK, status == GapLimitedLinearInterpolator.GAP_EXCEEDED],
        ["ok", "gap_exceeded"],
        "outside_hull",
    )
    out["fspl_term_db"] = free_space_term_db(out["range_m"].to_numpy())
    out["gain_db"] = out["power_db"] - out["fspl_term_db"] - out["rx_response_hat_db"] - TX_POWER_CORRECTION_DB
    return out


def build_body_frame_grid(samples: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Median-aggregate recovered gains per body-frame node, then fill empty nodes by gap-limited interpolation."""
    nodes = bin_median(
        samples["azimuth_deg"],
        samples["off_nadir_deg"],
        samples["gain_db"],
        GRID_OFF_NADIR_DEG.astype(float),
        1.0,
    )
    support = nodes[nodes["count"] >= PATTERN_MIN_SAMPLES]
    interp = GapLimitedLinearInterpolator(
        support["az_node"], support["polar_node"], support["median"], MAX_INTERP_GAP_DEG, AZ_WRAP_PAD_DEG
    )

    th_m, az_m = np.meshgrid(GRID_OFF_NADIR_DEG, GRID_AZIMUTH_DEG, indexing="ij")
    grid = pd.DataFrame({"azimuth_deg": az_m.ravel(), "off_nadir_deg": th_m.ravel()})
    val, status = interp(grid["azimuth_deg"].to_numpy(float), grid["off_nadir_deg"].to_numpy(float))
    counts = support.set_index(["az_node", "polar_node"])["count"]
    key = pd.MultiIndex.from_arrays([grid["azimuth_deg"].astype(float), grid["off_nadir_deg"].astype(float)])
    grid["sample_count"] = counts.reindex(key).fillna(0).astype(int).to_numpy()
    grid[GAIN_COLUMN] = val
    grid["normalized_gain_db"] = val - np.nanmax(val)
    grid = grid[["azimuth_deg", "off_nadir_deg", GAIN_COLUMN, "normalized_gain_db", "sample_count"]]

    measured = grid["sample_count"] > 0
    cell_status = np.select(
        [measured, status == GapLimitedLinearInterpolator.OK, status == GapLimitedLinearInterpolator.GAP_EXCEEDED],
        ["measured", "interpolated", "unsupported_gap"],
        "unsupported_outside_hull",
    )
    stats = {
        s: int((cell_status == s).sum())
        for s in ["measured", "interpolated", "unsupported_gap", "unsupported_outside_hull"]
    }
    stats["total_cells"] = len(grid)
    stats["unsupported_fraction"] = float(grid[GAIN_COLUMN].isna().mean())
    stats["samples_used_in_bins"] = int(support["count"].sum())
    return grid, stats


# =====================================================================================================
# Per-satellite preparation (validation + geometry)
# =====================================================================================================


def prepare_satellite(sat: str, data_dir: Path, r_gs: np.ndarray, enu: np.ndarray, need_heading: bool):
    rows, files = load_observations(data_dir, sat)
    rows["reject_reason"] = validate_rows(rows)
    by_reason, by_file = rejection_tables(rows)

    valid = rows[rows["reject_reason"] == ""].sort_values(["file", "unix_time", "time_s"]).reset_index(drop=True)
    knots, row_knot, n_conflicts = build_knots(valid, r_gs, enu)
    if need_heading:
        estimate_knot_headings(knots)
    geom = sample_geometry(valid, knots, row_knot)

    r_sat = geom["r_sat"]
    gs_az, gs_alt, rng = ecef_to_azalt_range(r_sat, r_gs, enu)
    valid["gs_az_deg"], valid["gs_alt_deg"], valid["range_m"] = gs_az, gs_alt, rng
    valid["off_boresight_deg"] = off_nadir_deg(r_sat, r_gs)
    valid["geometry_status"] = geom["geometry_status"]

    # Sanity: reconstructed orbit height above the ellipsoid (approx., using the geocentric latitude).
    rn = np.linalg.norm(knots.r, axis=1)
    sin_lat = knots.r[:, 2] / rn
    r_ell = WGS84_A_M * np.sqrt((1 - WGS84_E2) / (1 - WGS84_E2 * (1 - sin_lat**2)))
    height_km = (rn - r_ell) / 1000.0

    diag = {
        "files_total": len(files),
        "files_empty": int((files["n_rows"] == 0).sum()),
        "tracks_with_valid_rows": int(valid["file"].nunique()),
        "rows_read": len(rows),
        "rows_rejected": int(sum(by_reason.values())),
        "rows_rejected_by_reason": by_reason,
        "rows_valid": len(valid),
        "unique_position_knots": len(knots.t),
        "knot_geometry_conflicts": n_conflicts,
        "geometry_status_rows": valid["geometry_status"].value_counts().to_dict(),
        "tracks_with_inconsistent_time_offset": geom["bad_offset_files"],
        "reconstructed_height_km": {"min": float(height_km.min()), "max": float(height_km.max())},
        "off_nadir_range_deg": {
            "min": float(valid["off_boresight_deg"].min()),
            "max": float(valid["off_boresight_deg"].max()),
        },
    }

    if need_heading:
        valid["heading_status"] = geom["heading_status"]
        bf = body_frame_angles(r_sat, r_gs, geom["heading"])
        valid["azimuth_deg"], valid["off_nadir_deg"] = bf["azimuth_deg"], bf["off_nadir_deg"]
        valid.loc[~bf["valid"], "heading_status"] = "invalid"
        borrowed = knots.heading_status == "borrowed"
        diag["heading"] = {
            "velocity_frame": HEADING_VELOCITY_FRAME,
            "knot_status": pd.Series(knots.heading_status).value_counts().to_dict(),
            "knots_borrowed": int(borrowed.sum()),
            "max_borrow_distance_s": float(np.nanmax(knots.borrow_dt_s)) if borrowed.any() else 0.0,
            "rows_status": valid["heading_status"].value_counts().to_dict(),
            "rows_borrowed": int((valid["heading_status"] == "borrowed").sum()),
            "rows_invalid_excluded": int((valid["heading_status"] == "invalid").sum()),
        }
        diag["body_frame_orthonormality"] = bf["orthonormality"]
        both = bf["valid"]
        diag["off_nadir_equivalence_max_abs_diff_deg"] = float(
            np.abs(valid["off_nadir_deg"].to_numpy()[both] - valid["off_boresight_deg"].to_numpy()[both]).max()
        )

    return valid, files, by_file, diag


# =====================================================================================================
# Closure test
# =====================================================================================================


def closure_test(m24_samples: pd.DataFrame) -> dict:
    """Calibrate on even-numbered M2-4 tracks; recover G_t4 on odd-numbered tracks; compare to the reference."""
    even = m24_samples["obs_number"] % 2 == 0
    rx = build_receiver_map(m24_samples[even])
    test = m24_samples[~even]
    r_hat, status = rx.interpolator(test["gs_az_deg"].to_numpy(), test["gs_alt_deg"].to_numpy())
    ok = status == GapLimitedLinearInterpolator.OK
    recovered = test["power_db"].to_numpy() - test["fspl_term_db"].to_numpy() - r_hat
    resid = (recovered - test["ref_gain_db"].to_numpy())[ok]
    theta = test["off_boresight_deg"].to_numpy()[ok]

    # Same aggregation as the output pattern: median per (1 deg) off-nadir bin, then compare to the reference.
    th_bin = np.rint(theta).astype(int)
    per_bin = pd.Series(resid).groupby(th_bin).median()

    # Residual by M2-4 body-frame quadrant. An axisymmetric, nadir-pointing M2-4 antenna and a stable receiver
    # should give ~0 in every quadrant; a consistent quadrant offset means the reference assumption is violated
    # in the body frame, which bounds the accuracy of the transferred calibration.
    by_quadrant = {}
    if "azimuth_deg" in test:
        az = test["azimuth_deg"].to_numpy()[ok]
        quadrants = {
            "forward_315_45": (az >= 315) | (az < 45),
            "starboard_45_135": (az >= 45) & (az < 135),
            "aft_135_225": (az >= 135) & (az < 225),
            "port_225_315": (az >= 225) & (az < 315),
        }
        far = theta >= CLOSURE_QUADRANT_MIN_OFF_NADIR_DEG
        by_quadrant = {k: round(float(np.median(resid[m & far])), 3) for k, m in quadrants.items() if (m & far).any()}
    return {
        "residual_by_body_quadrant_db": by_quadrant,
        "calibration_tracks": int(m24_samples.loc[even, "file"].nunique()),
        "test_tracks": int(test["file"].nunique()),
        "test_rows": len(test),
        "test_rows_with_calibration": int(ok.sum()),
        "sample_residual_median_db": float(np.median(resid)),
        "sample_residual_mad_db": float(np.median(np.abs(resid - np.median(resid)))),
        "binned_residual_median_abs_db": float(per_bin.abs().median()),
        "binned_residual_max_abs_db": float(per_bin.abs().max()),
        "binned_residual_by_off_nadir_deg": {int(k): round(float(v), 3) for k, v in per_bin.items()},
    }


# =====================================================================================================
# Plotting
# =====================================================================================================


def _robust_limits(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, COLOR_PERCENTILES)
    return float(lo), float(hi)


def plot_body_frame_pattern(grid: pd.DataFrame, column: str, path: Path, title: str, cbar_label: str) -> None:
    """Polar heatmap: theta = body azimuth (0 = forward, clockwise toward +y_b), r = off-nadir (0 = centre)."""
    z = grid.pivot(index="off_nadir_deg", columns="azimuth_deg", values=column).to_numpy()
    az_edges = np.radians(np.r_[GRID_AZIMUTH_DEG - 0.5, GRID_AZIMUTH_DEG[-1] + 0.5])
    th_edges = np.clip(np.r_[GRID_OFF_NADIR_DEG - 0.5, GRID_OFF_NADIR_DEG[-1] + 0.5], 0, GRID_OFF_NADIR_DEG[-1])
    vmin, vmax = _robust_limits(z)

    fig, ax = plt.subplots(figsize=(9, 8.6), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # viewed from above the satellite looking toward nadir: +y_b (starboard) on the right
    missing = np.where(np.isnan(z), 1.0, np.nan)
    ax.pcolormesh(az_edges, th_edges, missing, cmap=ListedColormap([MISSING_COLOR]), vmin=0, vmax=1, shading="flat")
    mesh = ax.pcolormesh(
        az_edges, th_edges, np.ma.masked_invalid(z), cmap=GAIN_CMAP, vmin=vmin, vmax=vmax, shading="flat"
    )

    i_max = int(np.nanargmax(z))
    th_max, az_max = GRID_OFF_NADIR_DEG[i_max // z.shape[1]], GRID_AZIMUTH_DEG[i_max % z.shape[1]]
    ax.plot(
        np.radians(az_max), th_max, marker="+", color="#0b0b0b", markersize=12, markeredgewidth=1.6, linestyle="none"
    )

    ax.set_ylim(0, GRID_OFF_NADIR_DEG[-1])
    ax.set_yticks([15, 30, 45, 60])
    ax.set_yticklabels(
        ["15°", "30°", "45°", "60°"],
        color="#52514e",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    ax.set_rlabel_position(112)
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(
        ["0°\nforward +x_b", "45°", "90°\n+y_b", "135°", "180°\naft", "225°", "270°\n-y_b", "315°"], fontsize=9
    )
    ax.tick_params(axis="x", pad=8)
    ax.grid(color="#ffffff", alpha=0.5, linewidth=0.6)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.1, shrink=0.78, extend="both")
    cbar.set_label(cbar_label)
    ax.legend(
        handles=[
            Patch(facecolor=MISSING_COLOR, label="No supported estimate (NaN)"),
            plt.Line2D(
                [],
                [],
                marker="+",
                color="#0b0b0b",
                linestyle="none",
                markersize=10,
                label=f"Maximum: az {az_max}°, off-nadir {th_max}°",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(-0.12, -0.02),
        fontsize=8,
        frameon=False,
    )
    ax.set_title(title, fontsize=11, pad=22)
    fig.text(
        0.5,
        0.015,
        "Radius: off-nadir angle (0° = nadir at centre). Idealised body frame, viewed from above "
        "looking toward nadir. Colour range: 1st-99th percentile.",
        ha="center",
        fontsize=7.5,
        color="#52514e",
    )
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_receiver_map(grid: pd.DataFrame, path: Path) -> None:
    """Sky plot of R = C + G_r: theta = ground-station azimuth (N up, clockwise), r = 90 - elevation."""
    az = np.arange(0.0, 360.0, RX_MAP_NODE_STEP_DEG)
    el = np.arange(0.0, 90.0 + RX_MAP_NODE_STEP_DEG / 2, RX_MAP_NODE_STEP_DEG)
    z = grid.pivot(index="gs_alt_deg", columns="gs_az_deg", values="rx_response_db").to_numpy()
    az_edges = np.radians(np.r_[az - RX_MAP_NODE_STEP_DEG / 2, az[-1] + RX_MAP_NODE_STEP_DEG / 2])
    zen_edges = np.clip(90.0 - np.r_[el - RX_MAP_NODE_STEP_DEG / 2, el[-1] + RX_MAP_NODE_STEP_DEG / 2], 0, 90)
    vmin, vmax = _robust_limits(z)

    fig, ax = plt.subplots(figsize=(9, 8.6), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.pcolormesh(
        az_edges,
        zen_edges,
        np.where(np.isnan(z), 1.0, np.nan),
        cmap=ListedColormap([MISSING_COLOR]),
        vmin=0,
        vmax=1,
        shading="flat",
    )
    mesh = ax.pcolormesh(
        az_edges, zen_edges, np.ma.masked_invalid(z), cmap=RX_CMAP, vmin=vmin, vmax=vmax, shading="flat"
    )
    ax.set_ylim(0, 90)
    ax.set_yticks([30, 60, 90])
    ax.set_yticklabels(["60°", "30°", "0°"], color="#52514e", fontsize=8)
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "45°", "E", "135°", "S", "225°", "W", "315°"])
    ax.grid(color="#ffffff", alpha=0.5, linewidth=0.6)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.1, shrink=0.78, extend="both")
    cbar.set_label("Receiver response C + G_r (dB, arbitrary common offset)")
    ax.legend(
        handles=[Patch(facecolor=MISSING_COLOR, label="No supported estimate (NaN)")],
        loc="upper left",
        bbox_to_anchor=(-0.12, -0.02),
        fontsize=8,
        frameon=False,
    )
    ax.set_title(
        "Receiver directional response from Meteor-M N2-4 (ground-station frame)\n"
        "relative values; radius = 90° - elevation",
        fontsize=11,
        pad=22,
    )
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


# =====================================================================================================
# Diagnostics / metadata
# =====================================================================================================


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def format_summary(d: dict) -> str:
    s3, s4 = d["satellites"]["m2_3"], d["satellites"]["m2_4"]
    st1, st2, st3 = d["stage1_receiver_calibration"], d["stage2_m2_3_gain"], d["stage3_body_frame_pattern"]
    lines = [
        "M2-3 LRPT TRANSMIT ANTENNA PATTERN RECONSTRUCTION - DIAGNOSTICS",
        f"Generated {d['generated_utc']}. Gains in {GAIN_UNIT}, on the absolute scale of the supplied M2-4 reference "
        "(Ku_dB); see uncertainty_sources in m2_3_pattern_metadata.json.",
        "",
        "Inputs",
    ]
    for sat, s in (("M2-3", s3), ("M2-4", s4)):
        lines.append(
            f"  {sat}: {s['files_total']} files ({s['files_empty']} empty), {s['rows_read']:,} rows read, "
            f"{s['rows_rejected']:,} rejected, {s['rows_valid']:,} valid, {s['tracks_with_valid_rows']} tracks"
        )
        nonzero = {k: v for k, v in s["rows_rejected_by_reason"].items() if v}
        lines.append(f"        rejected by reason: {nonzero or 'none'}")
        lines.append(f"        geometry per row: {s['geometry_status_rows']}")
    lines += [
        "  Rejections per file: rejected_rows_by_file.csv",
        "",
        "Stage 1 - receiver calibration (M2-4, ground-station frame)",
        f"  M2-4 rows excluded (off-boresight outside reference pattern 0-{st1['reference_max_theta_deg']:.2f} deg): "
        f"{st1['rows_outside_reference_pattern']:,}",
        f"  M2-4 rows used: {st1['rows_used']:,} -> {st1['populated_nodes']:,} populated 1-deg nodes",
        f"  Coverage of 1-deg az/el grid (az 0-359, el 0-90): {st1['grid_coverage']}",
        f"  Supported region: az {st1['supported_az_range_deg']}, el {st1['supported_el_range_deg']} deg; "
        f"supported fraction by elevation band: {st1['supported_fraction_by_el_band']}",
        "",
        "Stage 2 - M2-3 gain recovery",
        f"  M2-3 rows without valid receiver calibration: {st2['rows_without_calibration']:,} "
        f"(outside hull {st2['rows_rx_outside_hull']:,}, "
        f"gap > {MAX_INTERP_GAP_DEG:g} deg {st2['rows_rx_gap_exceeded']:,})",
        f"  M2-3 rows with recovered gain: {st2['rows_with_gain']:,}",
        "",
        "Stage 3 - body-frame pattern",
        f"  Headings: {s3['heading']['rows_borrowed']:,} rows borrowed ({s3['heading']['knots_borrowed']} knots, "
        f"max distance {s3['heading']['max_borrow_distance_s']:.0f} s); "
        f"{s3['heading']['rows_invalid_excluded']:,} rows excluded for invalid heading",
        f"  Rows outside output grid (off-nadir > 62.5 deg): {st3['rows_outside_grid']:,}",
        f"  Rows used in final pattern: {st3['rows_used']:,} from {st3['tracks_used']} tracks",
        f"  Output cells: {st3['cells']['total_cells']:,} total; measured {st3['cells']['measured']:,}, "
        f"interpolated {st3['cells']['interpolated']:,}, unsupported (gap) {st3['cells']['unsupported_gap']:,}, "
        f"unsupported (outside hull) {st3['cells']['unsupported_outside_hull']:,} "
        f"-> {100 * st3['cells']['unsupported_fraction']:.1f}% NaN",
        f"  Gain range {st3['gain_range_db']} {GAIN_UNIT}; maximum (normalisation reference) at az "
        f"{st3['max_cell']['azimuth_deg']} deg, off-nadir {st3['max_cell']['off_nadir_deg']} deg: "
        f"{st3['max_cell']['sample_count']} samples from tracks {st3['max_cell']['tracks']}",
        "",
        "Interpolation",
        f"  {d['methods']['stage1_interpolation']}",
        f"  {d['methods']['stage3_interpolation']}",
        "",
        "Geometry checks",
        f"  Reconstructed orbit height (km): M2-3 {s3['reconstructed_height_km']}, "
        f"M2-4 {s4['reconstructed_height_km']}",
        f"  Body-frame orthonormality: {s3['body_frame_orthonormality']}",
        f"  max |off_nadir(body frame) - angle(q, nadir)|: {s3['off_nadir_equivalence_max_abs_diff_deg']:.2e} deg",
    ]
    if "closure_test" in d:
        c = d["closure_test"]
        lines += [
            "",
            "Closure test (calibrate on even M2-4 tracks, recover M2-4 on odd tracks, compare with reference)",
            f"  {c['test_rows_with_calibration']:,}/{c['test_rows']:,} test rows calibrated; per-sample residual "
            f"median {c['sample_residual_median_db']:+.3f} dB, MAD {c['sample_residual_mad_db']:.3f} dB",
            f"  1-deg off-nadir binned residual: median |r| {c['binned_residual_median_abs_db']:.3f} dB, "
            f"max |r| {c['binned_residual_max_abs_db']:.3f} dB",
            f"  Median residual by M2-4 body quadrant (off-nadir >= {CLOSURE_QUADRANT_MIN_OFF_NADIR_DEG:g} deg): "
            f"{c['residual_by_body_quadrant_db']} dB  (non-zero = M2-4 not axisymmetric/nadir-pointing as assumed)",
        ]
    return "\n".join(lines) + "\n"


def uncertainty_sources(diagnostics: dict) -> list[dict]:
    """The four systematic error sources of the gain transfer, with what this run's data can say about each."""
    c = diagnostics.get("closure_test", {})
    quad = c.get("residual_by_body_quadrant_db", {})
    overall = c.get("sample_residual_median_db")
    overall_txt = f"{overall:+.2f} dB" if overall is not None else "n/a"
    return [
        {
            "id": 1,
            "source": "Unequal transmit power at the antenna terminals",
            "description": "The transfer assumes both satellites deliver the same power to their antennas "
            "(6.5 W each, including spacecraft feed/cable losses).",
            "effect_on_m2_3": "Constant offset of the whole pattern, dB for dB; the shape is unaffected.",
            "current_handling": f"PT_M2_3_W = {PT_M2_3_W} W, PT_M2_4_W = {PT_M2_4_W} W -> correction "
            f"{TX_POWER_CORRECTION_DB:+.2f} dB.",
            "estimate_from_this_data": "Not observable: indistinguishable from a constant gain offset.",
            "how_to_reduce": "Put measured or telemetry transmitter output powers into PT_M2_3_W / PT_M2_4_W.",
        },
        {
            "id": 2,
            "source": "Different receive processing for the two signals",
            "description": "The receiver constant C must be identical for both satellites. The power estimator used "
            "different frequency offsets (M2-3 2180 Hz, M2-4 2980 Hz) through an uneven passband, so the fraction of "
            "signal power captured can differ between the satellites.",
            "effect_on_m2_3": "Constant offset of the whole pattern; a Doppler-dependent passband loss would also add "
            "forward/aft structure.",
            "estimate_from_this_data": "M2-4 hold-out residual forward "
            f"{quad.get('forward_315_45')} dB vs aft {quad.get('aft_135_225')} dB: no Doppler-dependent loss above "
            "~0.5 dB for M2-4. The constant part is not observable.",
            "how_to_reduce": "Measure the passband response at both frequency offsets and correct the captured power "
            "fraction per satellite.",
        },
        {
            "id": 3,
            "source": "Definition and shape of the M2-4 reference pattern",
            "description": "Whether Ku_dB is realised gain or directivity, dBi or dBic, and whether its off-boresight "
            f"shape is correct. Outputs are labelled {GAIN_UNIT} (GAIN_UNIT).",
            "effect_on_m2_3": "A unit/definition error is a constant offset. An error in the reference's theta-shape "
            "transfers 1:1 into M2-3's off-nadir dependence, because it is degenerate with the receiver's elevation "
            "response.",
            "estimate_from_this_data": "Not observable from these passes; the closure test checks only "
            "self-consistency.",
            "how_to_reduce": "Confirm the reference's provenance (measured or simulated, gain or directivity, "
            "polarisation).",
        },
        {
            "id": 4,
            "source": "In-flight M2-4 differs from the axisymmetric, nadir-pointing reference",
            "description": "Attitude offsets, antenna tilt or polarisation effects make the real M2-4 pattern "
            "depend on body azimuth.",
            "effect_on_m2_3": "Local, direction-dependent error: at each ground-station direction the receiver map "
            "absorbs M2-4's deviation averaged over the passes that saw that direction and passes it into M2-3 with "
            "the opposite sign.",
            "estimate_from_this_data": "M2-4 hold-out residual by body quadrant (off-nadir >= "
            f"{CLOSURE_QUADRANT_MIN_OFF_NADIR_DEG:g} deg): {quad} dB, i.e. roughly +/-1-2 dB locally; overall "
            f"offset {overall_txt}.",
            "how_to_reduce": f"{UNCERTAINTY_NOTEBOOK} estimates the M2-4 deviation map, propagates it into the M2-3 "
            "grid (m2_3_gain_uncertainty.csv) and adds per-cell bootstrap scatter.",
        },
    ]


def coverage_by_band(grid: pd.DataFrame, band_deg: float = 15.0) -> dict:
    supported = grid["rx_response_db"].notna()
    band = (grid["gs_alt_deg"] // band_deg * band_deg).astype(int)
    frac = supported.groupby(band).mean()
    return {f"{b}-{min(b + band_deg, 90):g}": round(float(v), 3) for b, v in frac.items()}


# =====================================================================================================
# Main
# =====================================================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--pattern", type=Path, default=M2_4_PATTERN_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--write-samples",
        action="store_true",
        help="also write per-sample M2-3 gains (m2_3_gain_samples.csv.gz, ~100 MB)",
    )
    args = parser.parse_args(argv)
    # Headless backend for script runs only, so importing this module from a notebook keeps inline plots.
    plt.switch_backend("Agg")
    data_dir, out_dir = args.data_dir.expanduser(), args.output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    r_gs = geodetic_to_ecef(GS_LAT_DEG, GS_LON_DEG, GS_ALT_M)
    enu = enu_basis(GS_LAT_DEG, GS_LON_DEG)
    ref = load_reference_pattern(args.pattern)

    print("Loading and validating M2-4 ...", flush=True)
    # M2-4 headings are only used by the closure test's body-frame quadrant check, not by the calibration.
    m24, files4, rej4, diag4 = prepare_satellite("m2_4", data_dir, r_gs, enu, need_heading=RUN_CLOSURE_TEST)
    print("Loading and validating M2-3 ...", flush=True)
    m23, files3, rej3, diag3 = prepare_satellite("m2_3", data_dir, r_gs, enu, need_heading=True)

    # ---- Stage 1 ------------------------------------------------------------------------------------
    print("Stage 1: receiver calibration ...", flush=True)
    m24 = receiver_response_samples(m24, ref)
    outside_ref = m24["ref_gain_db"].isna()
    m24_used = m24[~outside_ref]
    rx = build_receiver_map(m24_used)
    rx_grid = receiver_map_grid(rx)
    sup = rx_grid[rx_grid["rx_response_db"].notna()]
    stage1 = {
        "reference_first_sample_deg": ref.first_sample_deg,
        "reference_max_theta_deg": ref.max_theta_deg,
        "rows_outside_reference_pattern": int(outside_ref.sum()),
        "rows_used": len(m24_used),
        "populated_nodes": len(rx.nodes),
        "node_sample_count": {
            "min": int(rx.nodes["count"].min()),
            "median": float(rx.nodes["count"].median()),
            "max": int(rx.nodes["count"].max()),
        },
        "grid_coverage": rx_grid["status"].value_counts().to_dict(),
        "grid_supported_fraction": round(float(rx_grid["rx_response_db"].notna().mean()), 4),
        "supported_az_range_deg": [float(sup["gs_az_deg"].min()), float(sup["gs_az_deg"].max())],
        "supported_el_range_deg": [float(sup["gs_alt_deg"].min()), float(sup["gs_alt_deg"].max())],
        "supported_fraction_by_el_band": coverage_by_band(rx_grid),
    }

    # ---- Stage 2 ------------------------------------------------------------------------------------
    print("Stage 2: M2-3 gain recovery ...", flush=True)
    m23 = reconstruct_m23_gain(m23, rx)
    stage2 = {
        "rows_rx_outside_hull": int((m23["rx_map_status"] == "outside_hull").sum()),
        "rows_rx_gap_exceeded": int((m23["rx_map_status"] == "gap_exceeded").sum()),
        "rows_without_calibration": int((m23["rx_map_status"] != "ok").sum()),
        "rows_with_gain": int(m23["gain_db"].notna().sum()),
        "tx_power_correction_db": TX_POWER_CORRECTION_DB,
    }

    # ---- Stage 3 ------------------------------------------------------------------------------------
    print("Stage 3: body-frame pattern ...", flush=True)
    usable = m23["gain_db"].notna() & (m23["heading_status"] != "invalid")
    in_grid = np.rint(m23["off_nadir_deg"]) <= GRID_OFF_NADIR_DEG[-1]
    used = m23[usable & in_grid]
    grid, cell_stats = build_body_frame_grid(used)
    i_max = grid[GAIN_COLUMN].idxmax()
    az_max, th_max = int(grid.at[i_max, "azimuth_deg"]), int(grid.at[i_max, "off_nadir_deg"])
    in_max_cell = (np.mod(np.rint(used["azimuth_deg"]), 360) == az_max) & (np.rint(used["off_nadir_deg"]) == th_max)
    stage3 = {
        "rows_with_gain_but_invalid_heading": int(
            (m23["gain_db"].notna() & (m23["heading_status"] == "invalid")).sum()
        ),
        "rows_outside_grid": int((usable & ~in_grid).sum()),
        "rows_used": len(used),
        "tracks_used": int(used["file"].nunique()),
        "cells": cell_stats,
        "gain_unit": GAIN_UNIT,
        "gain_range_db": [round(float(grid[GAIN_COLUMN].min()), 3), round(float(grid[GAIN_COLUMN].max()), 3)],
        # The normalisation reference. Listed with its tracks because one anomalous pass can set it.
        "max_cell": {
            "azimuth_deg": az_max,
            "off_nadir_deg": th_max,
            "gain_db": round(float(grid.at[i_max, GAIN_COLUMN]), 3),
            "sample_count": int(grid.at[i_max, "sample_count"]),
            "tracks": sorted(used.loc[in_max_cell, "file"].unique().tolist()),
        },
    }

    # ---- Outputs ------------------------------------------------------------------------------------
    print("Writing outputs ...", flush=True)
    grid.to_csv(out_dir / PATTERN_CSV_NAME, index=False, float_format="%.4f")
    rx_grid.to_csv(out_dir / "receiver_response_map.csv", index=False, float_format="%.4f")
    pd.concat([rej3, rej4]).to_csv(out_dir / "rejected_rows_by_file.csv", index=False)
    pd.concat([files3, files4]).to_csv(out_dir / "input_files.csv", index=False)
    if args.write_samples:
        cols = [
            "file",
            "observation",
            "time",
            "unix_time",
            "time_s",
            "gs_az_deg",
            "gs_alt_deg",
            "range_m",
            "power_db",
            "fspl_term_db",
            "rx_response_hat_db",
            "rx_map_status",
            "gain_db",
            "geometry_status",
            "heading_status",
            "azimuth_deg",
            "off_nadir_deg",
        ]
        m23[cols].to_csv(out_dir / "m2_3_gain_samples.csv.gz", index=False, float_format="%.5f")

    plot_body_frame_pattern(
        grid,
        GAIN_COLUMN,
        out_dir / "m2_3_gain_polar.png",
        f"Meteor-M N2-3 LRPT transmit antenna gain ({GAIN_UNIT})\n"
        "absolute scale transferred from the M2-4 reference pattern",
        f"Gain ({GAIN_UNIT})",
    )
    plot_body_frame_pattern(
        grid,
        "normalized_gain_db",
        out_dir / "m2_3_normalized_gain_polar.png",
        "Meteor-M N2-3 LRPT transmit antenna: max-normalised gain\n(0 dB = grid maximum)",
        "Gain relative to grid maximum (dB)",
    )
    plot_receiver_map(rx_grid, out_dir / "receiver_response_map.png")

    diagnostics = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "satellites": {"m2_3": diag3, "m2_4": diag4},
        "stage1_receiver_calibration": stage1,
        "stage2_m2_3_gain": stage2,
        "stage3_body_frame_pattern": stage3,
        "methods": {
            "stage1_interpolation": interpolation_method_text(
                "Stage 1 (receiver map, ground-station frame)",
                f"{RX_MAP_NODE_STEP_DEG:g}-deg (azimuth 0-359, elevation 0-90) grid, evaluated at each M2-3 "
                "sample's own az/el",
            ),
            "stage3_interpolation": interpolation_method_text(
                "Stage 3 (M2-3 pattern, body frame)",
                "1-deg (azimuth 0-359, off-nadir 0-62) grid; measured nodes keep their median, empty nodes are "
                "interpolated",
            ),
        },
    }
    if RUN_CLOSURE_TEST:
        print("Closure test ...", flush=True)
        diagnostics["closure_test"] = closure_test(m24_used)

    metadata = {
        "description": "Reconstructed Meteor-M N2-3 LRPT transmitting antenna pattern in an idealised satellite "
        f"body frame. Gain in {GAIN_UNIT}: the unknown receiver calibration cancels, so the absolute scale is "
        "transferred from the supplied M2-4 reference pattern (Ku_dB). Accuracy is bounded by uncertainty_sources.",
        "gain_unit": GAIN_UNIT,
        "uncertainty_sources": uncertainty_sources(diagnostics),
        "outputs": {
            PATTERN_CSV_NAME: "Regular grid azimuth_deg 0..359 x off_nadir_deg 0..62 (1 deg). "
            f"{GAIN_COLUMN}: median of recovered per-sample gains at measured nodes (sample_count > 0), "
            "gap-limited linear interpolation at empty nodes (sample_count == 0), NaN where unsupported. "
            f"normalized_gain_db = {GAIN_COLUMN} - max({GAIN_COLUMN}).",
            "m2_3_gain_polar.png": f"Polar heatmap of {GAIN_COLUMN}.",
            "m2_3_normalized_gain_polar.png": "Polar heatmap of normalized_gain_db.",
            "receiver_response_map.csv/.png": "Stage-1 receiver response map R = C + G_r in the ground-station frame.",
            "rejected_rows_by_file.csv": "Rejected input rows per satellite, file and reason.",
            "input_files.csv": "Per-file row counts and time span.",
            "diagnostics.txt": "Human-readable summary of this metadata's diagnostics block.",
        },
        "model": {
            "equation": "p_ij = C + G_t,ij + G_r,j + L(d_ij);  L(d) = 20 log10(lambda / (4 pi d)), d in metres",
            "stage1": "R_j = p_4j - G_t4(theta_j) - L(d_4j) = C + G_r,j",
            "stage2": "G_t3 = p_3j - L(d_3j) - R_hat(az_3j, alt_3j) - 10 log10(PT_M2_3_W / PT_M2_4_W)",
            "notes": [
                "power_db is in arbitrary units; the receiver-chain calibration C cancels between the satellites, "
                "so no absolute received-power calibration is needed (gain-comparison method with M2-4 as the "
                f"reference antenna). The M2-3 gain inherits the absolute scale of Ku_dB ({GAIN_UNIT}).",
                "Receiver antenna is not assumed isotropic: its response is mapped in the ground-station frame.",
                "Polarisation mismatch neglected (both RHCP). Atmospheric/ionospheric losses not modelled.",
                "Non-positive power rows are rejected, never replaced. Median aggregation is applied to recovered "
                "gains, not raw power.",
                "Rows with power near the noise floor that were positive by chance are kept; the upstream "
                "dfs_filtered step removed rows below -50 dB and scintillation episodes.",
                "The M2-3 pattern is not assumed axisymmetric or nadir-pointing.",
            ],
        },
        "conventions": {
            "ground_station_frame": "az_deg clockwise from geodetic north in [0, 360); alt_deg above the WGS84 "
            "ellipsoid-normal horizon; ENU line of sight [cos(alt) sin(az), cos(alt) cos(az), sin(alt)].",
            "satellite_position": "r_sat = r_GS(WGS84 geodetic -> ECEF) + range_m * u_ECEF.",
            "body_frame": "Idealised (not measured attitude). z_b = geocentric nadir -r_sat/|r_sat|; x_b = direction "
            "of travel projected perpendicular to z_b; y_b = z_b x x_b (right-handed; starboard).",
            "body_azimuth": "azimuth_deg = atan2(q.y_b, q.x_b) in [0, 360); 0 = direction of travel, 90 = +y_b.",
            "off_nadir": "off_nadir_deg = atan2(sqrt(qx^2+qy^2), qz); 0 = nadir. Not the ground-station elevation.",
            "nadir_choice": "The task's printed nadir formula equals q (satellite->GS) and would give off_nadir = 0 "
            "everywhere; the geocentric nadir was used instead, as agreed with the user.",
            "m2_4_off_boresight": "Angle between satellite->GS and geocentric nadir (M2-4 boresight = nadir).",
            "plot_orientation": "Polar plots show the body frame viewed from above the satellite looking toward "
            "nadir: 0 deg (forward) at top, azimuth increasing clockwise toward +y_b.",
            "units": "Angles in degrees, distances in metres internally (range_km * 1000), gains in "
            f"{GAIN_UNIT} (normalized_gain_db and the receiver response map are relative dB).",
        },
        "geometry_timing": {
            "issue": "az/alt/range are given once per second at AOS + k s and were attached to 0.1 s power samples "
            "by floor(time_s), so geometry lags each sample by 0-0.9 s.",
            "handling": "ECEF position and travel direction linearly interpolated to AOS + time_s between the "
            f"surrounding per-second positions of the same track (gap <= {GEOMETRY_INTERP_MAX_GAP_S:g} s); "
            "otherwise the supplied geometry is used and the row is counted as fallback_supplied.",
            "enabled": INTERPOLATE_GEOMETRY_TO_SAMPLE_TIME,
        },
        "heading": {
            "method": "Per-second unique positions per track; central chord r[i+1]-r[i-1] if both steps <= "
            f"{HEADING_MAX_GAP_S:g} s, one-sided displacement at track ends or gaps, otherwise the time-nearest "
            "valid heading of the same track (borrowed, counted). Rows sharing a timestamp are collapsed first "
            "because adjacent raw rows have zero displacement.",
            "velocity_frame": HEADING_VELOCITY_FRAME,
            "note": "'ecef' uses velocity relative to the rotating Earth (as specified); this differs from the "
            "inertial orbital velocity direction by ~2-3.5 deg at these latitudes.",
        },
        "rejection_rules": {
            "precedence": REJECTION_REASONS,
            "non_finite_value": f"any of {NUMERIC_COLUMNS} missing or non-finite",
            "invalid_range": f"range_km <= {RANGE_KM_MIN_EXCLUSIVE:g} or > {RANGE_KM_MAX:g}",
            "invalid_angle": "az_deg outside [0, 360] or alt_deg outside [-90, 90]",
            "below_elevation_mask": f"alt_deg < {MIN_ELEVATION_DEG:g}",
            "non_positive_power": "power_linear <= 0",
            "power_linear_mismatch": f"|power_linear - (p_total - p_noise_scaled)| > max({POWER_LINEAR_ATOL:g}, "
            f"{POWER_LINEAR_RTOL:g} * |power_linear|)",
            "power_db_mismatch": f"|power_db - 10 log10(power_linear)| > {POWER_DB_ATOL:g} dB",
            "later_exclusions": "M2-4: off-boresight outside the reference pattern. M2-3: receiver map unsupported "
            "(outside hull or gap > limit); invalid heading; off_nadir rounds above 62 deg.",
        },
        "config": {
            "GAIN_UNIT": GAIN_UNIT,
            "PT_M2_3_W": PT_M2_3_W,
            "PT_M2_4_W": PT_M2_4_W,
            "PT_DBW": PT_DBW,
            "TX_POWER_CORRECTION_DB": TX_POWER_CORRECTION_DB,
            "FREQ_HZ": FREQ_HZ,
            "WAVELENGTH_M": WAVELENGTH_M,
            "GS_LAT_DEG": GS_LAT_DEG,
            "GS_LON_DEG": GS_LON_DEG,
            "GS_ALT_M": GS_ALT_M,
            "ellipsoid": "WGS84",
            "MAX_INTERP_GAP_DEG": MAX_INTERP_GAP_DEG,
            "HEADING_MAX_GAP_S": HEADING_MAX_GAP_S,
            "GEOMETRY_INTERP_MAX_GAP_S": GEOMETRY_INTERP_MAX_GAP_S,
            "RX_MAP_NODE_STEP_DEG": RX_MAP_NODE_STEP_DEG,
            "RX_MAP_MIN_SAMPLES": RX_MAP_MIN_SAMPLES,
            "PATTERN_MIN_SAMPLES": PATTERN_MIN_SAMPLES,
            "AZ_WRAP_PAD_DEG": AZ_WRAP_PAD_DEG,
            "MIN_ELEVATION_DEG": MIN_ELEVATION_DEG,
            "data_dir": data_dir,
            "reference_pattern": args.pattern,
        },
        "diagnostics": diagnostics,
    }
    (out_dir / "m2_3_pattern_metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2))
    summary = format_summary(diagnostics)
    (out_dir / "diagnostics.txt").write_text(summary)
    print()
    print(summary)
    print(f"Outputs written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
