"""
Find and analyze top k peaks in Fourier Transforms for each key
"""

import os
import numpy as np

from scipy.io import wavfile
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from json import JSONDecoder

DIR_IN = "../raw_data"
DIR_OUT = "../features"
KEYBOARD_TYPE = "mechanical"

# 2 * offset = n (length of FFT)
offset = 5000

num_peaks = 3

subset_labels = ["Space", "Backspace"]
cnt_labels = {}

# Output file
peaks_file = open(os.path.join(DIR_OUT, KEYBOARD_TYPE, "peaks.csv"), "w")
cols = "key"
for i in range(num_peaks):
    cols += ",peak_{},ampl_{}".format((i+1), (i+1))
cols += "\n"
peaks_file.write(cols)

cnt = 0

for f in os.listdir(os.path.join(DIR_IN, KEYBOARD_TYPE)):
    (basename, extension) = f.split(".")
    print("basename ", basename)

    # If it's a wav file, generate fft plot
    if extension == "wav":
        sample_rate, samples = wavfile.read(os.path.join(DIR_IN, KEYBOARD_TYPE, f))

        # Get corresponding ground truth JSON file
        labels_file = open(os.path.join(DIR_IN, KEYBOARD_TYPE, basename + ".json"))
        labels = JSONDecoder().decode(labels_file.read())
        for timestamp in labels:
            # the key that was pressed
            label = labels[timestamp]
            if label not in subset_labels:
               continue
            cnt_labels[label] = cnt_labels.get(label, 0) + 1
            print("label ", label)

            # timestamp is milliseconds since start of audio
            timestamp = int(timestamp)

            # Get the range of samples associated with the current key press
            sample_start = timestamp * sample_rate // 1000 - offset
            sample_end = timestamp * sample_rate // 1000 + offset

            # number of time samples or number of frequency bins
            n = sample_end - sample_start

            freq = np.fft.fftfreq(n, 1 / sample_rate)
            magnitude = np.fft.fft(samples[sample_start : sample_end])

            # only look at positive frequencies
            freq = freq[: n // 2]
            magnitude = magnitude[: n // 2]

            # Find the k highest peaks

            # 1. Stupid way
            # peak_freq = freq[np.argpartition(-magnitude, num_peaks)[: num_peaks]]

            # 2. Better way
            peaks_raw, props = find_peaks(magnitude, distance=200, height=max(magnitude)/50)
            print("len(peaks_raw) ", len(peaks_raw))

            # Sort the peaks in ascending order
            peaks = np.sort(peaks_raw)
            print("peak_freq ", peaks_raw)
            print()

            if len(peaks) < num_peaks:
                continue
            # Save peak data to output file
            peaks_file.write(label)
            print("magnitude[peaks ", np.real(magnitude[peaks]))
            max_mag = np.max(np.real(magnitude[peaks]))
            print("max_mag ", max_mag)

            max_freq = 5000
            for i in range(num_peaks):
                peaks_file.write(",{},{}".format(np.round(freq[peaks[i]] / max_freq, 3), np.round(np.real(magnitude[peaks[i]]) / max_mag, 3)))
            peaks_file.write("\n")

            # Plot fft
            ax = plt.gca()
            for peak in peaks_raw:
                ax.text(freq[peak], magnitude[peak], "x")
            plt.plot(freq, magnitude)
            plt.xlim([0, 6000])
            plt.ylim(bottom=0)
            plt.title("key = {}, time = {} ms, keyboard = {}".format(label, timestamp, KEYBOARD_TYPE))
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            if cnt < 6:
                plt.show()
            cnt += 1

        labels_file.close()

print("cnt_labels ", cnt_labels)
peaks_file.close()