from __future__ import annotations

import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from config.settings import settings
from storage.db import get_connection, get_latest_snapshot

KEPLER_INSERT_SQL = """
    INSERT INTO kepler (
        id, satellite_number, element_number, name, designator,
        epoch, inclination, right_ascension, eccentricity,
        argument_of_perigee, mean_anomaly, mean_motion,
        derivative_mean_motion, second_derivative_mean_motion,
        bstar_drag_term, revolutions_at_epoch
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _epoch_to_unix(epoch_str: str) -> float:
    """Convert an ISO-8601 epoch string (assumed UTC, per TIME_SYSTEM) to a
    unix timestamp, the same representation satdump stores in `epoch`."""
    dt = datetime.fromisoformat(epoch_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def update_satdump_kepler(norad_id: int) -> None:
    """Remove any existing kepler row(s) for `norad_id` and insert the
    latest orbital elements in their place."""
    with get_connection() as conn:
        snapshot = get_latest_snapshot(conn, norad_id)

    if snapshot is None:
        raise ValueError(f"No orbital snapshot found for NORAD ID {norad_id}")

    epoch_unix = _epoch_to_unix(snapshot["epoch"])
    row_id = f"{snapshot['norad_cat_id']}_{epoch_unix}"

    db_path = Path(settings.satdump.satdump_database_path_str).expanduser()

    sd_conn = sqlite3.connect(db_path)
    try:
        cur = sd_conn.cursor()
        cur.execute("DELETE FROM kepler WHERE satellite_number = ?", (norad_id,))
        cur.execute(
            KEPLER_INSERT_SQL,
            (
                row_id,
                snapshot["norad_cat_id"],
                snapshot["element_set_no"],
                snapshot["object_name"],
                snapshot["object_id"],
                epoch_unix,
                snapshot["inclination"],
                snapshot["ra_of_asc_node"],
                snapshot["eccentricity"],
                snapshot["arg_of_pericenter"],
                snapshot["mean_anomaly"],
                snapshot["mean_motion"],
                snapshot["mean_motion_dot"],
                snapshot["mean_motion_ddot"],
                snapshot["bstar"],
                snapshot["rev_at_epoch"],
            ),
        )
        sd_conn.commit()
    except Exception:
        sd_conn.rollback()
        raise
    finally:
        sd_conn.close()


def run_satdump(
    input_file: str,
    output_dir: str,
    f_s: int,
    run_path=settings.satdump.satdump_run_path_str,
    pipeline="meteor_m2-x_lrpt",
    mode="baseband",
    baseband_format="cs8",
):
    cmd = [
        "gobdump",
        "pipeline",
        pipeline,
        mode,
        input_file,
        output_dir,
        f"--samplerate={f_s}",
        f"--baseband_format={baseband_format}",
    ]
    return subprocess.run(cmd, cwd=run_path, text=True, check=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <norad_id>", file=sys.stderr)
        sys.exit(1)
    update_satdump_kepler(int(sys.argv[1]))
