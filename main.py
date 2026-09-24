from __future__ import annotations

import asyncio
import logging
import signal

from dotenv import load_dotenv

from config.loader import SatelliteLoader
from config.settings import settings
from orchestrator import Orchestrator
from scheduling.job_scheduler import JobScheduler
from storage.db import init_db

load_dotenv()

logger = logging.getLogger(__name__)

TARGET_SATELLITES_PATH = "config/target_satellites.json"


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    logger.info("Initializing database at %s", settings.database.sqlite_path)
    init_db()

    job_scheduler = JobScheduler(on_pass_due=lambda job: orchestrator.handle_pass_due(job))
    orchestrator = Orchestrator(job_scheduler)

    satellite_loader = SatelliteLoader(TARGET_SATELLITES_PATH)

    loop = asyncio.get_running_loop()

    def on_satellites_reloaded(satellites):
        asyncio.run_coroutine_threadsafe(job_scheduler.handle_satellites_reloaded(satellites), loop)

    satellite_loader.on_reload(on_satellites_reloaded)

    await orchestrator.recover_stale_jobs()

    job_scheduler.start()
    await job_scheduler.bootstrap(satellite_loader.get_satellites())
    satellite_loader.start()

    logger.info("Startup complete, running.")

    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()

    logger.info("Shutdown signal received, stopping...")
    satellite_loader.stop()
    job_scheduler.shutdown(wait=False)
    await orchestrator.shutdown()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
