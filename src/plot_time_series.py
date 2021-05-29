"""
Retrieve raw time series data and extract touch peaks
"""

import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from scipy.io import wavfile
from scipy.signal import find_peaks

DIR_IN = "../native_raw_data"
DIR_OUT = "../features"
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
                curr_timestamps.append(timestamp)

            timestamps.append(curr_timestamps)

    return np.array(res), sample_rate, np.array(timestamps)

def get_touch_fft(res, sample_rate, timestamps, key, features_file, touch_time=None, num_bins=None):

    for i in range(res.shape[0]):
        plt.title("key =  " + key)
        t = np.arange(0, res.shape[1] / sample_rate, 1.0 / sample_rate)
        peaks, props = find_peaks(res[i], height=500, distance=sample_rate/500)
        ax = plt.gca()
        if len(peaks) == 0:
            continue
        """
        if len(peaks) >= 3:
            ax.text(t[peaks[0]], 2000, "push", fontsize=12)
            ax.text(t[peaks[len(peaks) - 3]], 2000, "release", fontsize=12)

        for peak in peaks:
            ax.text(t[peak] - 0.001, res[i][peak] - 40, "o")
        """

        # We define the touch peak as the first peak over height 500 in the time series
        start = peaks[0] - int(sample_rate / 1000 * touch_time / 8)
        end = peaks[0] + int(sample_rate / 1000 * 7 * touch_time / 8)

        touch_samples = res[i][start : end]
        #touch_samples = np.hanning(len(touch_samples)) * touch_samples
        touch_t = t[start : end]

        if len(touch_samples) == 0:
            continue

        freq = np.fft.rfftfreq(len(touch_samples), 1 / sample_rate)
        magnitude = np.fft.rfft(touch_samples)

        # only look at positive frequencies
        freq = freq[: len(touch_samples) // 2]
        magnitude = np.abs(magnitude[: len(touch_samples) // 2])

        # Normalize fft
        magnitude /= np.max(magnitude)

        # Remove higher frequencies
        freq = freq[: num_bins]
        magnitude = magnitude[: num_bins]
        if len(freq) < num_bins:
            print("freq len is ", len(freq))
            print("skipping...")
            continue

        if i == 0:
            np.save(os.path.join(DIR_OUT, KEYBOARD_TYPE, "freq_bins_metadata.npy"), freq)

        features_file.write(key)
        for j in range(num_bins):
            features_file.write("," + str(magnitude[j]))
        features_file.write("\n")

        #print(i)
        #plt.xlim(right=10000)
        #plt.ylim(bottom=0)

        # plt.scatter(timestamps[i], np.zeros(len(timestamps[i])) + 1000, color='red')
        """
        if key == "w":
            plt.plot(touch_t, touch_samples)
            #plt.plot(freq, magnitude)
            #plt.plot(t, res[i])
            plt.show()
        """

if __name__ == "__main__":

    keys = ["space", "backspace", "a", "d", "f", "s", "e", "q", "w", "r", "g"]

    touch_time = 75
    max_freq = 6000
    fs = 44100

    # Number of frequency bins we store
    num_bins = int(touch_time * fs / 1000 * max_freq / fs)
    print("num_bins ", num_bins)

    features_file = open(os.path.join(DIR_OUT, KEYBOARD_TYPE, "touch_fft.csv"), 'w')
    features_file.write("key")

    for i in range(num_bins):
        features_file.write(",freq_bin_" + str(i + 1))
    features_file.write("\n")

    for key in keys:
        res, sample_rate, timestamps = get_time_series(key=key, num_samples=20000, entire=False)
        if sample_rate != fs:
            raise ValueError("sample_rate != " + fs)

        get_touch_fft(res, sample_rate, timestamps, key, features_file, touch_time=touch_time, num_bins=num_bins)

    features_file.close()
