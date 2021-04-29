"""
Find and analyze top k peaks in Fourier Transforms for each key
"""

import os
import numpy as np

from scipy.io import wavfile
from matplotlib import pyplot as plt
from json import JSONDecoder

DIR_IN = "raw_data"
DIR_OUT = "features"
KEYBOARD_TYPE = "membrane"

# 2 * offset = n (length of FFT)
offset = 5000

num_peaks = 4

subset_labels = ["a", "Space", "Backspace"]
cnt_labels = {}

# Output file
peaks_file = open(os.path.join(DIR_OUT, KEYBOARD_TYPE, "peaks.csv"), "w")
cols = "key"
for i in range(num_peaks):
    cols += ",peak_{}".format((i+1))
cols += "\n"
peaks_file.write(cols)

cnt = 0

for f in os.listdir(os.path.join(DIR_IN, KEYBOARD_TYPE)):
    (basename, extension) = f.split(".")

    # If it's a wav file, generate fft plot
    if extension == "wav":
        sample_rate, samples = wavfile.read(os.path.join(DIR_IN, KEYBOARD_TYPE, f))

        # Get corresponding ground truth JSON file
        labels_file = open(os.path.join(DIR_IN, KEYBOARD_TYPE, basename + ".json"))
        labels = JSONDecoder().decode(labels_file.read())
        for timestamp in labels:
            # the key that was pressed
            label = labels[timestamp]
            # if label not in subset_labels:
                # continue
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
            peak_freq = freq[np.argpartition(-magnitude, num_peaks)[: num_peaks]]

            # Sort the peaks in descending order
            peak_freq = -np.sort(-peak_freq)
            print("peak_freq ", peak_freq)
            print()

            # Save peak data to output file
            peaks_file.write(label)
            for i in range(num_peaks):
                peaks_file.write(",{}".format(np.round(peak_freq[i])))
            peaks_file.write("\n")

            # Plot fft
            plt.plot(freq, magnitude)
            plt.xlim([0, 10000])
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