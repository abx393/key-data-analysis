"""
Find and analyze top k peaks in Fourier Transforms for each key
"""

import os
import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from json import JSONDecoder

DIR_IN = "native_raw_data"
DIR_OUT = "features"
KEYBOARD_TYPE = "mechanical"

# 2 * offset = n (length of FFT)
offset = 5000

num_peaks = 3

subset_labels = ["space", "a", "backspace", "mouse_click"]
cnt_labels = {}

# Output file
peaks_file = open(os.path.join(DIR_OUT, KEYBOARD_TYPE, "peaks.csv"), "w")
cols = "key"
for i in range(num_peaks):
    cols += ",peak_{}".format((i+1), (i+1))
    # cols += ",peak_{},ampl_{}".format((i+1), (i+1))
cols += "\n"
peaks_file.write(cols)

cnt = 0

for f in os.listdir(os.path.join(DIR_IN, KEYBOARD_TYPE)):
    (basename, extension) = f.split(".")
    print("basename ", basename)

    # If it's a wav file, generate fft plot
    if extension == "wav":
        sample_rate, samples = wavfile.read(os.path.join(DIR_IN, KEYBOARD_TYPE, f))
        if len(samples.shape) == 2:
            samples = np.array(samples[:, 1])

        """
        # Get corresponding ground truth JSON file
        labels_file = open(os.path.join(DIR_IN, KEYBOARD_TYPE, basename + ".json"))
        labels = JSONDecoder().decode(labels_file.read())
        """

        # Get corresponding ground truth CSV file
        labels_file = open(os.path.join(DIR_IN, KEYBOARD_TYPE, basename + ".csv"))
        df = pd.read_csv(labels_file)
        labels_file.close()
        for i in range(len(df)):
            # the key that was pressed
            label = df.iloc[i, 0]

            # timestamp is seconds since start of audio recording
            timestamp = float(df.iloc[i, 1])

            if label not in subset_labels:
                continue
            cnt_labels[label] = cnt_labels.get(label, 0) + 1
            # print("label ", label)

            # Get the range of samples associated with the current key press
            sample_start = int(timestamp * sample_rate - offset)
            sample_end = int(timestamp * sample_rate + offset)

            # number of time samples or number of frequency bins
            n = sample_end - sample_start

            freq = np.fft.fftfreq(n, 1 / sample_rate)
            print("sample_start ", sample_start)
            print("sample_end ", sample_end)
            magnitude = np.fft.fft(samples[sample_start : sample_end])

            # only look at positive frequencies
            freq = freq[: n // 2]
            magnitude = magnitude[: n // 2]

            # Find the k highest peaks

            # 1. Stupid way
            # peak_freq = freq[np.argpartition(-magnitude, num_peaks)[: num_peaks]]

            # 2. Better way
            peaks_raw, props = find_peaks(magnitude, distance=150)
            """, height=max(magnitude)/50"""
            """ distance=150 """
            # print("len(peaks_raw) ", len(peaks_raw))

            # Sort the peaks in ascending order
            peaks = np.sort(peaks_raw)
            # print("peak_freq ", peaks_raw)
            print()

            if len(peaks) < num_peaks:
                continue
            # Save peak data to output file
            peaks_file.write(label)
            max_mag = np.max(np.real(magnitude[peaks]))

            # max_freq = 4000
            max_freq = 1
            for i in range(num_peaks):
                peaks_file.write(",{}".format(np.round(freq[peaks[i]] / max_freq, 3)))
            peaks_file.write("\n")

            # Plot fft
            ax = plt.gca()
            for peak in peaks_raw:
                ax.text(freq[peak], magnitude[peak], "x")
            plt.plot(freq, magnitude)
            plt.xlim([0, 6000])
            plt.ylim(bottom=0)
            plt.title("key = {}, time = {} s, keyboard = {}".format(label, timestamp, KEYBOARD_TYPE))
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.show()

        labels_file.close()

print("cnt_labels ", cnt_labels)
peaks_file.close()

df = pd.read_csv(os.path.join(DIR_OUT, KEYBOARD_TYPE, "peaks.csv"))

df.set_index("key", inplace=True)

legend = []
for key in subset_labels:
    print("Mean: key={}".format(key))
    print(df.loc[key, :].mean())
    print("Variance: key={}".format(key))
    print(df.loc[key, :].var())
    print()
    plt.title("Peak analysis: key={}".format(key))

    boxplot = df.loc[key, :].boxplot(column=["peak_1", "peak_2", "peak_3"])
    # legend.append("key={}".format(key))

    # plt.legend(legend)
    # plt.show()
