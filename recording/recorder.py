from __future__ import annotations

import argparse
import logging
import signal
import socket
import threading
from pathlib import Path

import numpy as np

from config.settings import settings
from recording.dsp import FrameProcessor, recv_exact, to_int8_iq  # noqa: F401 (to_int8_iq used inside dsp.py)
from storage.layout import paths_for

logger = logging.getLogger(__name__)

FRAME_BYTES = settings.rtl_tcp.frame_samples * 2

_recording_semaphore = threading.BoundedSemaphore(settings.max_concurrent_recordings)


def capture(
    norad_id: int,
    observation_dir: Path,
    duration_s: float,
    f_target: float,
    decim: int,
    host: str | None = None,
    port: int | None = None,
    stop_event: threading.Event | None = None,
):
    """
    Record one pass for `norad_id` at `f_target` Hz, decimated by `decim`,
    writing int8 I/Q samples to the baseband path for `observation_dir`
    (see storage/layout.py). Runs for `duration_s` seconds unless an
    external `stop_event` is supplied.
    """
    host = host or settings.rtl_tcp.host
    port = port or settings.rtl_tcp.port

    if not _recording_semaphore.acquire(blocking=False):
        logger.warning("[%s] skipped: %d recordings already in progress", norad_id, settings.max_concurrent_recordings)
        return

    owns_stop_event = stop_event is None
    stop_event = stop_event or threading.Event()
    timer = None
    if owns_stop_event:
        timer = threading.Timer(duration_s, stop_event.set)
        timer.start()

    processor = FrameProcessor(f_target, decim)
    outfile = paths_for(observation_dir, norad_id).baseband

    samples_written = 0

    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.settimeout(1.0)
            with open(outfile, "wb") as f:
                while not stop_event.is_set():
                    raw = recv_exact(sock, FRAME_BYTES, stop_event)
                    if raw is None:
                        break

                    u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                    iq = (u8[0::2] - 127.5) / 127.5 + 1j * (u8[1::2] - 127.5) / 127.5

                    decimated_i8 = processor.process(iq)
                    decimated_i8.tofile(f)
                    samples_written += len(decimated_i8)
    except OSError as e:
        logger.error("[%s] recording failed: %s", norad_id, e)
        return
    finally:
        if timer:
            timer.cancel()
        _recording_semaphore.release()

    out_rate = settings.rtl_tcp.sample_rate / decim
    logger.info(
        "[%s] saved %d samples (%.1f MB) at %.1f kS/s -> %s",
        norad_id,
        samples_written,
        samples_written * 2 / 1e6,
        out_rate / 1e3,
        outfile,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--norad-id", type=int, required=True)
    parser.add_argument("--observation-dir", required=True, type=Path)
    parser.add_argument("--f-center", type=float, required=True)
    parser.add_argument("--decimation", type=int, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_event.set())

    capture(
        norad_id=args.norad_id,
        observation_dir=args.observation_dir,
        duration_s=float("inf"),  # unused
        f_target=args.f_center,
        decim=args.decimation,
        stop_event=stop_event,
    )
