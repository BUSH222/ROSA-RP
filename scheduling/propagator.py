from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from skyfield.api import EarthSatellite, load, wgs84

from config.settings import settings

if TYPE_CHECKING:
    from core.models import OmmSnapshot

logger = logging.getLogger(__name__)

ts = load.timescale()


def _qth():
    return wgs84.latlon(settings.site.lat, settings.site.lon)


def satellite_from_snapshot(snapshot: OmmSnapshot) -> EarthSatellite:
    """Build a skyfield EarthSatellite from a stored OMM snapshot's raw fields."""
    snapshot_raw_uppercase = {}
    for k, v in snapshot.raw.items():
        snapshot_raw_uppercase[k.upper()] = v
    snapshot_raw_uppercase["EPOCH"] = snapshot.epoch
    return EarthSatellite.from_omm(ts, snapshot_raw_uppercase)


def find_passes(satellite: EarthSatellite, start: datetime, end: datetime, min_elevation_deg: float = 0.0):
    """(aos, los) UTC datetime tuples for `satellite` in [start, end].
    Incomplete intervals at either edge (already up at `start`, still up at
    `end`) are dropped - can't schedule a clean recording for those.
    """
    t0 = ts.from_datetime(start)
    t1 = ts.from_datetime(end)

    t, events = satellite.find_events(_qth(), t0, t1, altitude_degrees=min_elevation_deg)

    intervals: list[list] = []
    for ti, event in zip(t, events, strict=True):
        if event == 0:  # rise
            intervals.append([ti.utc_datetime(), None])
        elif event == 2:  # set
            if intervals and intervals[-1][1] is None:
                intervals[-1][1] = ti.utc_datetime()
            else:
                logger.debug("%s: set event with no matching rise, dropping", satellite.name)

    return [(aos, los) for aos, los in intervals if los is not None]


def next_pass(
    satellite: EarthSatellite,
    after: datetime,
    min_elevation_deg: float = 0.0,
    step_hours: float = 12.0,
    max_hours: float = 96.0,
):
    window_hours = step_hours
    while window_hours <= max_hours:
        passes = find_passes(satellite, after, after + timedelta(hours=window_hours), min_elevation_deg)
        if passes:
            return passes[0]
        window_hours += step_hours
    logger.warning("No pass found for %s within %sh of %s", satellite.name, max_hours, after)
    return None


def initial_schedule(satellites: list[EarthSatellite], horizon_hours: float = 12.0, min_elevation_deg: float = 0.0):
    """Convenience for CLI/inspection: next pass per satellite as a plain
    dict {satnum: (aos, los)}. job_scheduler.py does its own per-satellite
    version of this since it also needs to write Jobs to the DB.
    """
    now = datetime.now(UTC)
    schedule: dict[int, tuple[datetime, datetime]] = {}
    for satellite in satellites:
        passes = find_passes(satellite, now, now + timedelta(hours=horizon_hours), min_elevation_deg)
        if not passes:
            passes = [next_pass(satellite, now, min_elevation_deg, step_hours=horizon_hours)]
            passes = [p for p in passes if p]
        if passes:
            schedule[satellite.model.satnum] = passes[0]
    return schedule
