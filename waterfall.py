import numpy as np
from PIL import Image
from scipy.fft import fft
from scipy.signal.windows import nuttall

## Output image
IMG_DEFAULT_WIDTH = 2**12  # px
IMG_AXIS = False

## FFT
FFT_WINDOW = 2**15
FFT_AVG = 1
FFT_MAX_DIFF_DB = 30
FFT_SCALING_BATCHES = 5

## Input
INPUT_SAMPLE_RATE = 300e3
INPUT_CENTER_FREQ = 138e6
INPUT_FORMAT = "int16"  # "int16" or "int8"


def save_grayscale_png(output_file, rows):
    image_array = np.stack(rows, axis=0)
    img = Image.fromarray(image_array, mode="L")
    img.save(output_file)


def process(input_file, output_file):
    window = nuttall(FFT_WINDOW)
    rows = []
    input_dtype = np.int8 if INPUT_FORMAT == "int8" else np.int16

    # calculate the min and max power ceiling
    min_dbs = []

    with open(input_file, "rb") as f:
        for _ in range(FFT_SCALING_BATCHES):
            raw = np.fromfile(f, dtype=input_dtype, count=FFT_WINDOW * 2)
            i = raw[0::2].astype(np.float32)
            q = raw[1::2].astype(np.float32)
            spectrum = np.fft.fftshift(fft(i * window + 1j * q * window, workers=-1))
            spectrum_compressed = np.mean(spectrum.reshape(-1, FFT_WINDOW // IMG_DEFAULT_WIDTH), axis=1)
            power = spectrum_compressed.real**2 + spectrum_compressed.imag**2
            power_db = 10 * np.log10(power + 1e-12)
            min_dbs.append(power_db)

    min_db = sum(min_dbs) / len(min_dbs) - 5
    max_db = min_db + FFT_MAX_DIFF_DB

    with open(input_file, "rb") as f:
        while True:
            raw = np.fromfile(f, dtype=input_dtype, count=FFT_WINDOW * 2)

            if len(raw) < FFT_WINDOW * 2:
                break

            i = raw[0::2].astype(np.float32)
            q = raw[1::2].astype(np.float32)

            spectrum = np.fft.fftshift(fft(i * window + 1j * q * window, workers=-1))

            # Compress spectrum to target width:
            spectrum_compressed = np.mean(spectrum.reshape(-1, FFT_WINDOW // IMG_DEFAULT_WIDTH), axis=1)
            power = spectrum_compressed.real**2 + spectrum_compressed.imag**2
            power_db = 10 * np.log10(power + 1e-12)

            row = np.clip((power_db - min_db) / (max_db - min_db) * 255, 0, 255)
            rows.append(row.astype(np.uint8))

    save_grayscale_png(output_file, rows)
    return len(rows)


if __name__ == "__main__":
    filename = "/Users/tedvtorov/Desktop/57166_20260910T180503Z.iq"
    out = "out.png"
    process(filename, out)
