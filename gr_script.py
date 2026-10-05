#!/usr/bin/env python3

"""
Measure average signal power in the central part of a 150 kS/s IQ recording.

Input:
    Interleaved signed 8-bit IQ:
        I0 Q0 I1 Q1 I2 Q2 ...

Processing:

    150 kS/s IQ
        |
        v
    Doppler correction
        |
        v
    anti-alias low-pass filter
        |
        v
    decimate by 2
        |
        v
    75 kS/s complex IQ
        |
        v
    |IQ|^2
        |
        v
    accumulate exactly 75,000 samples
        |
        v
    1-second average
        |
        v
    10 log10(power)

Output CSV:

    second,power_linear,power_db

The CSV is driven by the processed sample stream. There is no wall-clock
sleep and no probe polling.

For every complete 75,000-sample block after decimation, one row is written.

Because the input is a recorded file, processing may be faster or slower
than wall-clock time. "One second" means one second of recorded signal time.

The resulting power is in arbitrary digital units because the IQ samples
are assumed to be signed 8-bit samples. It is not directly dBm without
calibration.
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


class OneSecondPowerWriter(gr.sync_block):
    """
    Consume |IQ|^2 samples and write one CSV row per recorded second.

    GNU Radio may give work() arbitrary-sized chunks, so this block keeps
    a running sum and sample count until exactly samples_per_second samples
    have been accumulated.
    """

    def __init__(self, samples_per_second, output_file, scale=1.0):
        self.samples_per_second = int(samples_per_second)
        self.scale = float(scale)

        gr.sync_block.__init__(
            self,
            name="One Second Power Writer",
            in_sig=[np.float32],
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
                "second",
                "power_linear",
                "power_db",
            ]
        )

        self._csv_file.flush()

        self._sum_power = 0.0
        self._sample_count = 0
        self._second = 0
        self._closed = False

    @property
    def measurements_written(self):
        return self._second

    @property
    def partial_samples(self):
        return self._sample_count

    def _write_measurement(self):
        power_linear = self._sum_power / self.samples_per_second

        power_db = 10.0 * math.log10(power_linear) if power_linear > 0.0 else float("-inf")

        self._writer.writerow(
            [
                self._second,
                f"{power_linear:.6f}",
                f"{power_db:.6f}",
            ]
        )

        self._csv_file.flush()

        print(
            f"{self._second:6d} s   power = {power_linear:12.4f}   {power_db:9.3f} dB",
            flush=True,
        )

        self._second += 1
        self._sum_power = 0.0
        self._sample_count = 0

    def work(self, input_items, output_items):
        data = input_items[0]
        n = len(data)

        pos = 0

        while pos < n:
            remaining = self.samples_per_second - self._sample_count

            take = min(
                remaining,
                n - pos,
            )

            chunk = data[pos : pos + take]

            self._sum_power += float(
                np.sum(
                    chunk,
                    dtype=np.float64,
                )
            )

            self._sample_count += take
            pos += take

            if self._sample_count == self.samples_per_second:
                self._write_measurement()

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
        lpf_cutoff=40000.0,
        lpf_transition=2500.0,
        scale=1000.0,
    ):

        gr.top_block.__init__(
            self,
            "75 kHz Signal Power Measurement",
        )

        self.input_rate = float(input_rate)

        # ------------------------------------------------------------
        # Decimation
        # ------------------------------------------------------------

        self.decimation = 2

        self.output_rate = self.input_rate / self.decimation

        if not self.output_rate.is_integer():
            raise ValueError("Input sample rate must produce an integer output sample rate after decimation by 2")

        self.samples_per_second = round(self.output_rate)

        # ------------------------------------------------------------
        # IQ file
        #
        # Signed 8-bit interleaved IQ:
        #
        #   I0 Q0 I1 Q1 I2 Q2 ...
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
            128,
        )

        # ------------------------------------------------------------
        # Doppler correction
        #
        # Must operate before decimation, at the original
        # 150 kS/s sample rate.
        # ------------------------------------------------------------

        self.doppler = satellites.doppler_correction(
            doppler_file,
            self.input_rate,
            0,
        )

        # ------------------------------------------------------------
        # Anti-alias low-pass filter
        #
        # Decimation by 2 changes:
        #
        #   150 kS/s -> 75 kS/s
        #
        # The output Nyquist frequency is therefore 37.5 kHz.
        #
        # A practical filter cannot have a perfect brick-wall at
        # exactly 37.5 kHz. We use:
        #
        #   passband cutoff   = 35 kHz
        #   transition width  = 2.5 kHz
        #
        # so the stopband begins at 37.5 kHz.
        #
        # This preserves approximately the central 70 kHz with
        # proper anti-alias protection.
        # ------------------------------------------------------------
        self.keep_one_in_n = blocks.keep_one_in_n(
            gr.sizeof_gr_complex,
            2,
        )

        self.lpf = filter.fir_filter_ccf(
            self.decimation,
            firdes.low_pass(
                1.0,
                self.input_rate,
                lpf_cutoff,
                lpf_transition,
                window.WIN_HAMMING,
                6.76,
            ),
        )

        # ------------------------------------------------------------
        # Instantaneous power
        #
        # |I + jQ|^2 = I^2 + Q^2
        # ------------------------------------------------------------

        self.mag_squared = blocks.complex_to_mag_squared(1)

        # ------------------------------------------------------------
        # Exactly one recorded second per CSV row.
        #
        # At 75 kS/s:
        #
        #   75,000 samples = 1 second
        #
        # This block is sample-driven, so no time.sleep() is needed.
        # ------------------------------------------------------------

        self.power_writer = OneSecondPowerWriter(
            self.samples_per_second,
            output_file,
            scale=scale,
        )

        # ------------------------------------------------------------
        # Connections
        # ------------------------------------------------------------

        self.connect(
            self.file_source,
            self.interleaved_to_complex,
        )

        self.connect(
            self.interleaved_to_complex,
            self.doppler,
        )

        self.connect(
            self.doppler,
            self.lpf,
        )

        self.connect(
            self.lpf,
            self.mag_squared,
        )

        self.connect(
            self.mag_squared,
            self.power_writer,
        )

    def close(self):
        self.power_writer.close()


def main():

    parser = argparse.ArgumentParser(
        description=("Measure average signal power in the central part of a 150 kHz IQ recording.")
    )

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
        type=float,
        default=1.0,
        help="Multiplier on the mean power. 1000 reproduces the reference "
        "flowgraph (moving_average_ff length=1000, scale=1 -> sum, +30 dB). "
        "Use 1 for true mean power. (default: 1000)",
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
            scale=args.scale,
        )

        print()
        print("75 kHz signal power measurement")
        print("--------------------------------")
        print(f"IQ file:        {args.iq}")
        print(f"Doppler file:   {args.doppler}")
        print(f"Input rate:     {args.sample_rate:.0f} S/s")
        print(f"Output rate:    {tb.output_rate:.0f} S/s")
        print("LPF cutoff:     35 kHz")
        print("LPF transition: 2.5 kHz")
        print(f"Samples/row:    {tb.samples_per_second} (exactly 1 recorded second)")
        print()
        print("Processing file; no wall-clock sleep is used.")
        print()

        tb.start()

        # Wait until the finite IQ file has been completely processed.
        # No polling and no sleep are required.
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

            partial = tb.power_writer.partial_samples

            tb.close()

            print()
            print(f"Wrote {measurements} complete measurements to {args.output}")

            if partial:
                print(f"Ignored final incomplete second: {partial}/{tb.samples_per_second} samples")


if __name__ == "__main__":
    main()
