#!/usr/bin/env python3

"""
Measure noise-subtracted signal power in a 150 kS/s IQ recording.

Input:
    Interleaved signed 8-bit IQ:
        I0 Q0 I1 Q1 I2 Q2 ...

Processing:

    int8 IQ file -> complex (scale 1/127)
        |
        v
    Doppler correction
        |
        v
    Rotator (static SDR frequency offset, --freq-offset)
        |
        +--> signal LPF  (58 kHz, 8 kHz transition) -> |x|^2 -> integrate(N) -> /N = P_tot
        |
        +--> noise BPF   (64-72 kHz, 4 kHz trans.)  -> |x|^2 -> integrate(N) -> /N = P_noise_raw

    P_signal = P_tot - K * P_noise_raw          (K = 14.5, hardcoded)
    power_db = 10 log10(P_signal)

N is --scale (default 15000 samples = 0.1 s at 150 kS/s).

Output CSV (one row per N samples, driven by the sample stream, no
wall-clock sleep):

    time_s,p_total,p_noise_scaled,power_linear,power_db

    p_total         mean |x|^2 in the signal filter (signal + noise)
    p_noise_scaled  K * mean |x|^2 in the noise filter (noise estimate in
                    the signal band)
    power_linear    p_total - p_noise_scaled
    power_db        10 log10(power_linear), -inf if power_linear <= 0

The resulting power is in arbitrary digital units (dBFS-like). It is not
dBm without calibration.
"""

import argparse
import csv
import math
import signal

import numpy as np
import pmt
import satellites
from gnuradio import blocks, filter, gr
from gnuradio.fft import window
from gnuradio.filter import firdes

# Ratio of noise power in the signal filter to noise power in the noise filter.
NOISE_K = 14.5


class PowerWriter(gr.sync_block):
    """
    Consume three synchronous float streams (one value per averaging window)
    and write one CSV row per value:

        input 0: P_tot           (signal + noise, signal filter)
        input 1: K * P_noise_raw (noise estimate scaled to the signal band)
        input 2: P_tot - K * P_noise_raw
    """

    def __init__(self, output_file, seconds_per_row):
        self.seconds_per_row = float(seconds_per_row)

        gr.sync_block.__init__(
            self,
            name="Power Writer",
            in_sig=[np.float32, np.float32, np.float32],
            out_sig=[],
        )

        self._csv_file = open(  # noqa
            output_file,
            "w",
            newline="",
            encoding="utf-8",
        )

        self._writer = csv.writer(self._csv_file)

        self._writer.writerow(
            [
                "time_s",
                "p_total",
                "p_noise_scaled",
                "power_linear",
                "power_db",
            ]
        )

        self._csv_file.flush()

        self._rows = 0
        self._closed = False

    @property
    def measurements_written(self):
        return self._rows

    def work(self, input_items, output_items):
        p_tot = input_items[0]
        p_noise = input_items[1]
        p_sig = input_items[2]

        n = min(len(p_tot), len(p_noise), len(p_sig))

        for i in range(n):
            power_linear = float(p_sig[i])

            power_db = 10.0 * math.log10(power_linear) if power_linear > 0.0 else float("-inf")

            self._writer.writerow(
                [
                    f"{self._rows * self.seconds_per_row:.4f}",
                    f"{float(p_tot[i]):.9f}",
                    f"{float(p_noise[i]):.9f}",
                    f"{power_linear:.9f}",
                    f"{power_db:.6f}",
                ]
            )

            self._rows += 1

        self._csv_file.flush()

        return n

    def close(self):
        if self._closed:
            return

        self._closed = True
        self._csv_file.close()


class PowerMeasurement(gr.top_block):
    def __init__(
        self,
        iq_file,
        doppler_file,
        input_rate=150000.0,
        output_file="power.csv",
        avgs=15000,
        frequency_offset=2180.0,
    ):

        gr.top_block.__init__(
            self,
            "Noise-subtracted Signal Power Measurement",
        )

        self.samp_rate = float(input_rate)
        self.avgs = int(avgs)
        self.frequency_offset = float(frequency_offset)

        if self.avgs < 1:
            raise ValueError("--scale (averaging length) must be >= 1")

        # ------------------------------------------------------------
        # Filters
        # ------------------------------------------------------------

        self.LPF = firdes.low_pass(
            1.0,
            self.samp_rate,
            58e3,
            8e3,
            window.WIN_HAMMING,
            6.76,
        )

        self.BPF = firdes.band_pass(
            1.0,
            self.samp_rate,
            64e3,
            72e3,
            4e3,
            window.WIN_HAMMING,
            6.76,
        )

        # ------------------------------------------------------------
        # IQ file (signed 8-bit interleaved IQ), read once.
        # ------------------------------------------------------------

        self.file_source = blocks.file_source(
            gr.sizeof_char,
            iq_file,
            False,
            0,
            0,
        )

        self.file_source.set_begin_tag(pmt.PMT_NIL)

        self.interleaved_to_complex = blocks.interleaved_char_to_complex(
            False,
            127,
        )

        # ------------------------------------------------------------
        # Doppler correction, then static frequency offset removal
        # ------------------------------------------------------------

        self.doppler = satellites.doppler_correction(
            doppler_file,
            self.samp_rate,
            0,
        )

        self.freqshift = blocks.rotator_cc(2.0 * math.pi * self.frequency_offset / self.samp_rate)

        # ------------------------------------------------------------
        # Signal branch: LPF -> |x|^2 -> integrate -> mean
        # ------------------------------------------------------------

        self.signal_fir = filter.fir_filter_ccc(1, self.LPF)
        self.signal_fir.declare_sample_delay(0)

        self.signal_mag2 = blocks.complex_to_mag_squared(1)
        self.signal_integrate = blocks.integrate_ff(self.avgs, 1)
        self.signal_mean = blocks.multiply_const_ff(1.0 / self.avgs)

        # ------------------------------------------------------------
        # Noise branch: BPF -> |x|^2 -> integrate -> mean -> * K
        # ------------------------------------------------------------

        self.noise_fir = filter.fir_filter_ccc(1, self.BPF)
        self.noise_fir.declare_sample_delay(0)

        self.noise_mag2 = blocks.complex_to_mag_squared(1)
        self.noise_integrate = blocks.integrate_ff(self.avgs, 1)
        self.noise_mean = blocks.multiply_const_ff(1.0 / self.avgs)
        self.noise_k = blocks.multiply_const_ff(NOISE_K)

        # ------------------------------------------------------------
        # P_signal = P_tot - K * P_noise_raw
        # ------------------------------------------------------------

        self.subtract = blocks.sub_ff(1)

        # ------------------------------------------------------------
        # CSV writer: one row per averaging window
        # ------------------------------------------------------------

        self.power_writer = PowerWriter(
            output_file,
            self.avgs / self.samp_rate,
        )

        # ------------------------------------------------------------
        # Connections
        # ------------------------------------------------------------

        self.connect(self.file_source, self.interleaved_to_complex)
        self.connect(self.interleaved_to_complex, self.doppler)
        self.connect(self.doppler, self.freqshift)

        self.connect(self.freqshift, self.signal_fir)
        self.connect(self.freqshift, self.noise_fir)

        self.connect(self.signal_fir, self.signal_mag2)
        self.connect(self.signal_mag2, self.signal_integrate)
        self.connect(self.signal_integrate, self.signal_mean)

        self.connect(self.noise_fir, self.noise_mag2)
        self.connect(self.noise_mag2, self.noise_integrate)
        self.connect(self.noise_integrate, self.noise_mean)
        self.connect(self.noise_mean, self.noise_k)

        # subtract: in0 - in1 = P_tot - K * P_noise_raw
        self.connect(self.signal_mean, (self.subtract, 0))
        self.connect(self.noise_k, (self.subtract, 1))

        # writer inputs: P_tot, K * P_noise_raw, difference
        self.connect(self.signal_mean, (self.power_writer, 0))
        self.connect(self.noise_k, (self.power_writer, 1))
        self.connect(self.subtract, (self.power_writer, 2))

    def close(self):
        self.power_writer.close()


def main():

    parser = argparse.ArgumentParser(description=("Measure noise-subtracted signal power in a 150 kHz IQ recording."))

    parser.add_argument(
        "--iq",
        required=True,
        help="Input interleaved signed 8-bit IQ file",
    )

    parser.add_argument(
        "--doppler",
        required=True,
        help="GNU Radio Doppler correction file",
    )

    parser.add_argument(
        "--output",
        default="power.csv",
        help="Output CSV file",
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        default=150000.0,
        help="Input sample rate (default: 150000 S/s)",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=15000,
        help="Averaging length in samples (integrate-and-dump size, 'avgs' in "
        "the flowgraph). One output row per this many samples; 15000 samples "
        "at 150 kS/s = 0.1 s. (default: 15000)",
    )

    parser.add_argument(
        "--freq-offset",
        type=float,
        default=2180.0,
        help="Static frequency offset in Hz to remove after Doppler correction "
        "(rotator phase increment = 2*pi*offset/sample_rate). "
        "(default: 2180)",
    )

    args = parser.parse_args()

    tb = None

    def sig_handler(sig, frame):
        if tb is not None:
            tb.stop()

    signal.signal(
        signal.SIGINT,
        sig_handler,
    )

    signal.signal(
        signal.SIGTERM,
        sig_handler,
    )

    try:
        tb = PowerMeasurement(
            iq_file=args.iq,
            doppler_file=args.doppler,
            input_rate=args.sample_rate,
            output_file=args.output,
            avgs=args.scale,
            frequency_offset=args.freq_offset,
        )

        tb.start()

        # Wait until the finite IQ file has been completely processed.
        tb.wait()

    except KeyboardInterrupt:
        if tb is not None:
            tb.stop()
            tb.wait()

    finally:
        if tb is not None:
            # Make sure GNU Radio has completely stopped before
            # closing the CSV file.
            tb.stop()
            tb.wait()

            measurements = tb.power_writer.measurements_written

            tb.close()

            print()
            print(f"Wrote {measurements} measurements to {args.output}")


if __name__ == "__main__":
    main()
