from __future__ import annotations

from functools import cache

import numpy as np
from scipy import signal

from config.settings import settings


@cache
def design_decimating_filter(decim: int) -> np.ndarray:
    """Design a lowpass FIR filter (float32 taps) for decimating by `decim`."""
    numtaps = max(64, 8 * decim)
    if numtaps % 2 == 0:
        numtaps += 1
    taps = signal.firwin(numtaps, cutoff=1.0 / decim, window="hamming")
    return taps.astype(np.float32)


def to_int16_iq(decimated_c64: np.ndarray, scale: int = 32767) -> np.ndarray:
    """Convert interleaved complex64 IQ samples to clipped int16 I/Q pairs."""
    interleaved = decimated_c64.view(np.float32)
    scaled = np.round(interleaved * scale)
    return np.clip(scaled, -32768, 32767).astype(np.int16)


class FrameProcessor:
    def __init__(self, f_target: float, decim: int):
        self.freq_offset = f_target - settings.rtl_tcp.f_center
        self.sample_rate = settings.rtl_tcp.sample_rate
        self.decim = decim
        self.taps = design_decimating_filter(decim)

        self._phase = 0.0
        self._zi = np.zeros(len(self.taps) - 1, dtype=np.complex64)

    def process(self, iq: np.ndarray) -> np.ndarray:
        """Mix, filter, and decimate one frame of raw IQ; returns int16 I/Q pairs."""
        w = 2 * np.pi * self.freq_offset / self.sample_rate
        n_idx = np.arange(len(iq))

        osc = np.exp(-1j * w * n_idx + 1j * self._phase)
        self._phase = (self._phase + w * len(iq)) % (2 * np.pi)

        mixed = (iq * osc).astype(np.complex64)
        filtered, self._zi = signal.lfilter(self.taps, 1.0, mixed, zi=self._zi)
        decimated = filtered[:: self.decim].astype(np.complex64)
        return to_int16_iq(decimated)


def recv_exact(sock, nbytes: int, stop_event) -> bytes | None:
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
