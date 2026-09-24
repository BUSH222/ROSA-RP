from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import Job
    from storage.layout import ObservationPaths

logger = logging.getLogger(__name__)


@dataclass
class RecordingOutcome:
    success: bool
    returncode: int | None = None
    error: str | None = None


class ProcessSupervisor:
    """Each recording runs as its own OS process (so a DSP crash can't take
    down the scheduler, and pid lets us detect orphans on restart).
    """

    def __init__(self, python_exe: str | None = None, term_grace_s: float = 10.0):
        self.python_exe = python_exe or sys.executable
        self.term_grace_s = term_grace_s

    async def spawn(self, job: Job, obs_dir: ObservationPaths) -> asyncio.subprocess.Process:
        argv = [
            self.python_exe,
            "-m",
            "recording.recorder",
            "--observation-dir",
            str(obs_dir.root),
            "--job-id",
            job.id,
            "--norad-id",
            str(job.norad_id),
            "--f-center",
            str(job.f_center),
            "--decimation",
            str(job.decimation),
        ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        logger.info("[%s] spawned recorder pid=%s", job.norad_id, proc.pid)
        return proc

    async def wait(
        self,
        proc: asyncio.subprocess.Process,
        deadline: datetime,
        stop_event: asyncio.Event,
    ) -> RecordingOutcome:
        """Blocks until, whichever comes first:
        - LOS deadline reached -> SIGTERM (expected, clean stop)
        - stop_event set externally (job cancelled) -> SIGTERM
        - process exits on its own (crash) -> inspect returncode
        """
        timeout = max((deadline - datetime.now(UTC)).total_seconds(), 0)

        proc_task = asyncio.create_task(proc.wait())
        stop_task = asyncio.create_task(stop_event.wait())

        done, pending = await asyncio.wait(
            {proc_task, stop_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if proc_task in done:
            stop_task.cancel()
            returncode = proc_task.result()
            if returncode == 0:
                return RecordingOutcome(success=True, returncode=0)
            stderr = await self._drain_stderr(proc)
            return RecordingOutcome(success=False, returncode=returncode, error=stderr or f"exited {returncode}")

        for t in pending:
            t.cancel()
        return await self._terminate(proc)

    async def _terminate(self, proc: asyncio.subprocess.Process) -> RecordingOutcome:
        if proc.returncode is not None:
            return RecordingOutcome(success=proc.returncode == 0, returncode=proc.returncode)

        logger.info("Sending SIGTERM to pid=%s", proc.pid)
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return RecordingOutcome(success=True, returncode=proc.returncode)

        try:
            await asyncio.wait_for(proc.wait(), timeout=self.term_grace_s)
        except TimeoutError:
            logger.warning("pid=%s did not exit within %ss, sending SIGKILL", proc.pid, self.term_grace_s)
            proc.kill()
            await proc.wait()

        return RecordingOutcome(success=True, returncode=proc.returncode)

    @staticmethod
    async def _drain_stderr(proc: asyncio.subprocess.Process) -> str | None:
        if proc.stderr is None:
            return None
        data = await proc.stderr.read()
        return data.decode(errors="replace").strip() or None
