"""
Retrieve raw time series data
"""

import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from scipy.io import wavfile
from scipy.signal import find_peaks

DIR_IN = "../native_raw_data"
KEYBOARD_TYPE = "mechanical"

def get_time_series(key=None, path=DIR_IN, num_samples=44000, entire=False):
    res = []
    timestamps = []
    for f in os.listdir(os.path.join(path, KEYBOARD_TYPE)):
        (basename, extension) = f.split(".")

        # If it's a wav file, generate fft plot
        if extension == "wav":
            sample_rate, samples = wavfile.read(os.path.join(path, KEYBOARD_TYPE, f))
            curr_timestamps = []

            # print("samples.shape ", samples.shape)
            if len(samples.shape) == 2:
                samples = np.array(samples[:, 1])

            if entire:
                res.append(samples)

            # Get corresponding ground truth CSV file
            labels_file = open(os.path.join(path, KEYBOARD_TYPE, basename + ".csv"))
            df = pd.read_csv(labels_file)
            labels_file.close()
            for i in range(len(df)):
                # the key that was pressed
                label = df.iloc[i, 0]
                if key != None and label != key:
                    continue

                # timestamp is seconds since start of audio recording
                timestamp = float(df.iloc[i, 1])

                # Get the range of samples associated with the current key press
                sample_start = int(timestamp * sample_rate - num_samples / 2)
                sample_end = int(timestamp * sample_rate + num_samples / 2)
                if sample_start < 0 or sample_end > len(samples):
                    continue

                if not entire:
                    res.append(samples[sample_start : sample_end])
                    print("len ", len(samples[sample_start: sample_end]))
                curr_timestamps.append(timestamp)

            timestamps.append(curr_timestamps)

    return np.array(res), sample_rate, np.array(timestamps)

if __name__ == "__main__":
    key = "space"
    res, sample_rate, timestamps = get_time_series(key=key, num_samples=20000, entire=False)

    for i in range(res.shape[0]):
        plt.title("key =  " + key)
        t = np.arange(0, res.shape[1] / sample_rate, 1.0 / sample_rate)
        peaks, props = find_peaks(res[i], height=500, distance=sample_rate/500)
        ax = plt.gca()
        if len(peaks) == 0:
            continue
        if len(peaks) >= 3:
            ax.text(t[peaks[0]], 2000, "push", fontsize=12)
            ax.text(t[peaks[len(peaks) - 3]], 2000, "release", fontsize=12)

        for peak in peaks:
            ax.text(t[peak] - 0.001, res[i][peak] - 40, "o")

        plt.plot(t[peaks[0] : peaks[2]] - 0.005, res[i][peaks[0] : peaks[2]])
        # plt.scatter(timestamps[i], np.zeros(len(timestamps[i])) + 1000, color='red')
        plt.show()