import socket
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from scipy import signal

from tracker import TARGET_SATELLITES, get_schedule

RTL_TCP_HOST = "localhost"
RTL_TCP_PORT = 1234
SAMPLE_RATE = 2_400_000
F_CENTER = 137e6

FRAME_SAMPLES = 32768
FRAME_BYTES = FRAME_SAMPLES * 2

OUTPUT_DIR = Path("recordings")
MAX_CONCURRENT_RECORDINGS = 5

OUTPUT_DIR.mkdir(exist_ok=True)

# Filtering and decimation


def design_decimating_filter(decim):
    numtaps = max(64, 8 * decim)
    if numtaps % 2 == 0:
        numtaps += 1
    taps = signal.firwin(numtaps, cutoff=1.0 / decim, window="hamming")
    return taps.astype(np.float32)


FILTER_TAPS = {decim: design_decimating_filter(decim) for _f_target, decim in TARGET_SATELLITES.values()}


# Sample conversion
def to_int16_iq(decimated_c64, scale=32767):
    interleaved = decimated_c64.view(np.float32)
    scaled = np.round(interleaved * scale)
    return np.clip(scaled, -32768, 32767).astype(np.int16)


# Sockets


def recv_exact(sock, nbytes, stop_event):
    buf = bytearray()
    while len(buf) < nbytes:
        if stop_event.is_set():
            return None
        try:
            chunk = sock.recv(nbytes - len(buf))
        except TimeoutError:
            continue
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# Capture


def capture(norad_id, duration_s, f_target, decim, host=RTL_TCP_HOST, port=RTL_TCP_PORT):
    freq_offset = f_target - F_CENTER
    taps = FILTER_TAPS[decim]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    outfile = OUTPUT_DIR / f"{norad_id}_{timestamp}.iq"

    stop_event = threading.Event()
    timer = threading.Timer(duration_s, stop_event.set)
    timer.start()

    phase = 0.0
    zi = np.zeros(len(taps) - 1, dtype=np.complex64)

    n_idx = np.arange(FRAME_SAMPLES)
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

                    # translate
                    osc = np.exp(-1j * 2 * np.pi * freq_offset / SAMPLE_RATE * n_idx + 1j * phase)
                    phase = (phase + 2 * np.pi * freq_offset / SAMPLE_RATE * FRAME_SAMPLES) % (2 * np.pi)
                    mixed = (iq * osc).astype(np.complex64)

                    # filter + decimate
                    filtered, zi = signal.lfilter(taps, 1.0, mixed, zi=zi)
                    decimated = filtered[::decim].astype(np.complex64)
                    decimated_i16 = to_int16_iq(decimated)
                    decimated_i16.tofile(f)

                    samples_written += len(decimated_i16)
    except OSError as e:
        print(f"[{norad_id}] recording failed: {e}")
        return
    finally:
        timer.cancel()

    out_rate = SAMPLE_RATE / decim
    print(
        f"[{norad_id}] saved {samples_written} samples "
        f"({samples_written * 8 / 1e6:.1f} MB) at {out_rate / 1e3:.1f} kS/s -> {outfile}"
    )


def schedule_recordings(schedules, scheduler):
    for norad_id, windows in schedules.items():
        if norad_id not in TARGET_SATELLITES:
            print(f"[{norad_id}] no entry in TARGET_SATELLITES, skipping all windows")
            continue
        f_target, decim = TARGET_SATELLITES[norad_id]

        for start, end in windows:
            duration_s = (end - start).total_seconds()
            if duration_s <= 0:
                print(f"[{norad_id}] skipping invalid window {start} -> {end}")
                continue
            scheduler.add_job(
                capture,
                trigger="date",
                run_date=start,
                args=[norad_id, duration_s, f_target, decim],
                id=f"{norad_id}_{start.isoformat()}",
                misfire_grace_time=30,
            )
            print(f"[{norad_id}] scheduled {start} -> {end} ({duration_s:.0f}s, decim={decim})")


if __name__ == "__main__":
    schedules = get_schedule(TARGET_SATELLITES)

    executors = {"default": ThreadPoolExecutor(max_workers=MAX_CONCURRENT_RECORDINGS)}
    sched = BackgroundScheduler(executors=executors)

    schedule_recordings(schedules, sched)
    sched.start()

    print("Scheduler started. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt, SystemExit:
        sched.shutdown()
