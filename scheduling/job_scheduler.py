from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from core.job_state import JobState
from core.models import Job, OmmSnapshot
from scheduling import propagator
from scheduling.omm_updater import update_omm
from storage.db import (
    create_job,
    get_connection,
    get_latest_snapshot,
    invalidate_overdue_pending_jobs,
    list_jobs,
)

logger = logging.getLogger(__name__)

OMM_REFRESH_INTERVAL_HOURS = 24
PassDueCallback = Callable[[Job], Awaitable[None]]


class JobScheduler:
    def __init__(self, on_pass_due: PassDueCallback):
        self._on_pass_due = on_pass_due
        self._scheduler = AsyncIOScheduler(timezone=UTC)
        self._satellites: dict[int, object] = {}  # norad_id: Satellite

    def start(self):
        self._scheduler.start()

    def shutdown(self, wait: bool = True):
        self._scheduler.shutdown(wait=wait)

    def set_target_satellites(self, satellites):
        self._satellites = {sat.id: sat for sat in satellites}

    async def bootstrap(self, satellites):
        """Startup: refresh OMMs, then ensure every target satellite has an
        active (pending/recording) job, seeding one if not.
        """
        self.set_target_satellites(satellites)
        await self.refresh_omms()

        with get_connection() as conn:
            invalidated = invalidate_overdue_pending_jobs(conn)
            if invalidated:
                logger.warning("Invalidated %s overdue pending job(s)", invalidated)

            active = {j["norad_id"] for j in list_jobs(conn, state=JobState.PENDING.value)}
            active |= {j["norad_id"] for j in list_jobs(conn, state=JobState.RECORDING.value)}

        for sat in satellites:
            if sat.id in active:
                logger.info("Satellite %s already has an active job, skipping seed", sat.id)
                continue
            await self._schedule_next_pass(sat)

        self._scheduler.add_job(
            self.refresh_omms,
            trigger=IntervalTrigger(hours=OMM_REFRESH_INTERVAL_HOURS),
            id="omm_refresh",
            replace_existing=True,
        )

    async def refresh_omms(self):
        try:
            await asyncio.to_thread(update_omm, list(self._satellites.values()))
        except Exception:
            logger.exception("OMM refresh failed")

    async def _schedule_next_pass(self, sat, after: datetime | None = None):
        after = after or datetime.now(UTC)

        with get_connection() as conn:
            snapshot_row = get_latest_snapshot(conn, sat.id)
        if snapshot_row is None:
            logger.warning("No OMM snapshot yet for %s (%s); cannot schedule", sat.name, sat.id)
            return None

        snapshot = OmmSnapshot.from_row(snapshot_row)
        earth_satellite = propagator.satellite_from_snapshot(snapshot)
        pass_ = await asyncio.to_thread(propagator.next_pass, earth_satellite, after)
        if pass_ is None:
            logger.error("Could not find a next pass for %s (%s)", sat.name, sat.id)
            return None

        aos, los = pass_
        with get_connection() as conn:
            job = create_job(
                conn,
                norad_id=sat.id,
                aos=aos.isoformat(),
                los=los.isoformat(),
                f_center=sat.frequency,
                decimation=sat.decimation_factor,
                omm_snapshot_id=snapshot_row["id"],
            )
        self._schedule_job(job, aos)
        logger.info("Scheduled %s (%s): %s -> %s", sat.name, sat.id, aos, los)
        return job

    def _schedule_job(self, job: Job, aos: datetime):
        self._scheduler.add_job(
            self._fire,
            trigger=DateTrigger(run_date=aos),
            id=f"job:{job.id}",
            args=[job],
            replace_existing=True,
            misfire_grace_time=60,
        )

    async def _fire(self, job: Job):
        try:
            await self._on_pass_due(job)
        except Exception:
            logger.exception("on_pass_due callback failed for job %s", job.id)

    async def on_observation_finished(self, job: Job):
        """Orchestrator calls this once `job` reaches a terminal state, to
        queue that satellite's next pass (no data salvage on failure - point 7)."""
        sat = self._satellites.get(job.norad_id)
        if sat is None:
            logger.warning("Job %s finished for untracked satellite %s", job.id, job.norad_id)
            return
        await self._schedule_next_pass(sat, datetime.now(UTC))

    async def handle_satellites_reloaded(self, satellites):
        """Wire this to SatelliteLoader.on_reload. Newly added satellites get
        seeded with a first pass; satellites removed from the file are just
        untracked here - any job already scheduled for them is left alone."""
        old_ids = set(self._satellites)
        self.set_target_satellites(satellites)
        for sat in satellites:
            if sat.id not in old_ids:
                await self._schedule_next_pass(sat)
