import argparse
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from skyfield.api import EarthSatellite, load, wgs84


def get_omm_snapshot(snapshot_id, db_path="satcap.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            "SELECT * FROM omm_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()

        if row is None:
            raise ValueError(f"OMM snapshot not found: {snapshot_id}")

        columns = conn.execute("PRAGMA table_info(omm_snapshots)").fetchall()

        column_names = [column[1] for column in columns]

        return list(zip(column_names, list(row), strict=True))
    finally:
        conn.close()


def parse_utc_time(value):
    """Convert an ISO-8601 timestamp or UNIX timestamp to a UTC datetime."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)

    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))

    dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)

    return dt


def satellite_from_omm(omm):
    """
    Construct a Skyfield EarthSatellite from OMM orbital elements.

    `omm` can be either:
        - a dict
        - a list of (key, value) tuples
    """
    if not isinstance(omm, dict):
        omm = dict(omm)

    ts = load.timescale()

    # OMM epoch is UTC and has no timezone suffix in your data.
    datetime.fromisoformat(omm["epoch"]).replace(tzinfo=UTC)

    # Skyfield's EarthSatellite constructor expects TLE lines, so construct
    # the satellite directly from the OMM elements using the SGP4 model.
    sat = EarthSatellite(
        omm["tle_line1"],
        omm["tle_line2"],
        omm["object_name"],
        ts,
    )

    return sat, ts


def doppler_track(
    metadata,
    omm,
    station_lat,
    station_lon,
    station_alt_m=0.0,
    rate_hz=10.0,
):
    """
    Generate a Doppler time/frequency series for a satellite pass.

    Returns a pandas DataFrame with:
        time        UTC datetime
        unix_time   UNIX timestamp
        range_km    satellite range from station
        range_rate_m_s  radial velocity
        doppler_hz  Doppler offset from nominal center frequency
    """

    # Normalize OMM
    if not isinstance(omm, dict):
        omm = dict(omm)

    # ------------------------------------------------------------------
    # Input metadata
    # ------------------------------------------------------------------

    aos = parse_utc_time(metadata["aos"])
    los = parse_utc_time(metadata["los"])
    f_nom = float(metadata["f_center"])

    # ------------------------------------------------------------------
    # Satellite
    # ------------------------------------------------------------------

    sat, ts = satellite_from_omm(omm)

    # Ground station
    site = wgs84.latlon(
        station_lat,
        station_lon,
        elevation_m=station_alt_m,
    )

    # ------------------------------------------------------------------
    # Time vector
    # ------------------------------------------------------------------

    duration = (los - aos).total_seconds()

    n_samples = round(duration * rate_hz) + 1

    times = [aos + timedelta(seconds=i / rate_hz) for i in range(n_samples)]

    t_sf = ts.from_datetimes(times)

    # ------------------------------------------------------------------
    # Relative satellite state
    # ------------------------------------------------------------------

    rel = (sat - site).at(t_sf)

    r_km = np.asarray(rel.position.km)
    v_km_s = np.asarray(rel.velocity.km_per_s)

    # r and v have shape (3, N)
    r_norm = np.linalg.norm(r_km, axis=0)

    dot_rv = np.einsum("ij,ij->j", r_km, v_km_s)

    # Positive = satellite moving away from station
    range_rate_km_s = dot_rv / r_norm
    range_rate_m_s = range_rate_km_s * 1000.0

    # ------------------------------------------------------------------
    # Doppler
    # ------------------------------------------------------------------

    c = 299_792_458.0

    # Same convention as your original script:
    #
    # approaching  -> positive Doppler
    # receding     -> negative Doppler
    #
    doppler_hz = -(range_rate_m_s / c) * f_nom

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    unix_time = np.array([t.timestamp() for t in times])

    df = pd.DataFrame(
        {
            "time": times,
            "unix_time": unix_time,
            "range_km": r_norm,
            "range_rate_m_s": range_rate_m_s,
            "doppler_hz": doppler_hz,
        }
    )

    return df


def write_doppler_file(
    observation_dir,
    db_path,
    output_filename="doppler_track.txt",
    station_lat=55.5269,
    station_lon=37.0888,
    station_alt_m=127.0,
    rate_hz=10.0,
):
    observation_dir = Path(observation_dir).expanduser().resolve()
    db_path = Path(db_path).expanduser().resolve()

    metadata_path = observation_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    if not db_path.exists():
        raise FileNotFoundError(f"Missing database: {db_path}")

    with metadata_path.open() as file:
        metadata = json.load(file)

    omm = get_omm_snapshot(
        metadata["omm_snapshot_id"],
        db_path=db_path,
    )

    track = doppler_track(
        metadata=metadata,
        omm=omm,
        station_lat=station_lat,
        station_lon=station_lon,
        station_alt_m=station_alt_m,
        rate_hz=rate_hz,
    )

    output_path = observation_dir / output_filename

    output_track = track.copy()
    output_track["elapsed_s"] = output_track["unix_time"] - output_track["unix_time"].iloc[0]

    output_track[["elapsed_s", "doppler_hz"]].to_csv(
        output_path,
        sep=" ",
        header=False,
        index=False,
        float_format="%.6f",
    )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a Doppler track text file for a satellite observation.")

    parser.add_argument(
        "-i",
        "--input",
        dest="observation_dir",
        required=True,
        help="Observation directory containing metadata.json",
    )

    parser.add_argument(
        "--db",
        dest="db_path",
        default=Path(__file__).resolve().parent / "satcap.db",
        type=Path,
        help="Path to satcap.db. Defaults to satcap.db next to this script.",
    )

    args = parser.parse_args()

    output_path = write_doppler_file(
        observation_dir=args.observation_dir,
        db_path=args.db_path,
    )

    print(f"Wrote Doppler track to {output_path}")


if __name__ == "__main__":
    main()
