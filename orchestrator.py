from __future__ import annotations

import asyncio
import logging
from datetime import datetime

import hooks.observation_start
import hooks.post_observation  # noqa: F401
from core.events import Event, registry
from core.job_state import JobState, validate_transition
from core.models import Job
from process_supervisor import ProcessSupervisor
from storage.db import get_connection, list_jobs, update_job_state
from storage.layout import new_observation_dir, paths_for

logger = logging.getLogger(__name__)


class Orchestrator:
    """Owns a single job's lifecycle from AOS to a terminal state.
    Wire as: JobScheduler(on_pass_due=orchestrator.handle_pass_due)
    """

    def __init__(self, job_scheduler, supervisor: ProcessSupervisor | None = None):
        self._job_scheduler = job_scheduler
        self._supervisor = supervisor or ProcessSupervisor()
        self._running: dict[str, asyncio.Task] = {}

    async def recover_stale_jobs(self):
        """Startup: a job stuck in 'recording' means the app died mid-pass.
        No partial pass is salvageable, so fail it and move on (point 7)."""
        with get_connection() as conn:
            stale = list_jobs(conn, state=JobState.RECORDING.value)
        for row in stale:
            job = Job.from_row(row)
            logger.warning("Stale recording job %s (pid=%s) at startup, marking failed", job.id, job.pid)
            self._transition(job, JobState.FAILED, failure_reason="orphaned at startup")
            await self._job_scheduler.on_observation_finished(job)

    async def handle_pass_due(self, job: Job):
        """Called by JobScheduler exactly at AOS."""
        task = asyncio.create_task(self._run_observation(job))
        self._running[job.id] = task
        task.add_done_callback(lambda t, jid=job.id: self._running.pop(jid, None))

    async def _run_observation(self, job: Job):
        stop_event = asyncio.Event()
        los = datetime.fromisoformat(job.los)

        obs_root = new_observation_dir(job.norad_id)
        job.observation_dir = str(obs_root)
        obs_dir = paths_for(obs_root, job.norad_id)

        try:
            proc = await self._supervisor.spawn(job, obs_dir)
        except Exception:
            logger.exception("Failed to spawn recorder for job %s", job.id)
            self._transition(job, JobState.FAILED, failure_reason="spawn failed")
            await self._job_scheduler.on_observation_finished(job)
            return

        self._transition(job, JobState.RECORDING, pid=proc.pid, observation_dir=job.observation_dir)

        start_hook_tasks = registry.emit(Event.OBSERVATION_START, job, stop_event)

        outcome = await self._supervisor.wait(proc, deadline=los, stop_event=stop_event)

        stop_event.set()
        await asyncio.gather(*start_hook_tasks, return_exceptions=True)

        if not outcome.success:
            self._transition(job, JobState.FAILED, failure_reason=outcome.error or "recording failed")
            await self._job_scheduler.on_observation_finished(job)
            return

        post_hook_tasks = registry.emit(Event.POST_OBSERVATION, job, stop_event)
        await asyncio.gather(*post_hook_tasks, return_exceptions=True)

        self._transition(job, JobState.COMPLETED)
        await self._job_scheduler.on_observation_finished(job)

    def _transition(self, job: Job, new_state: JobState, **kwargs):
        validate_transition(job.state, new_state.value)
        job.state = new_state.value
        with get_connection() as conn:
            update_job_state(conn, job.id, new_state.value, **kwargs)

    async def shutdown(self):
        """Let in-flight observations run to their own deadline/signal rather
        than cancelling mid-pass."""
        if self._running:
            await asyncio.gather(*self._running.values(), return_exceptions=True)
