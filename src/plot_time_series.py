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
KEYBOARD_TYPE = "Dell"

def get_time_series(key=None, path=DIR_IN, num_samples=44000, entire=False):
    """
    Gets raw audio samples in time domain for all instances of the key press of `key`
    in all WAV files in the directory at `path`.

    :param key: the key whose time series audio is desired
    :param path: file path of directory containing WAV files
    :param num_samples:
    :param entire: whether to return the entire file's audio samples (not just the key requested)
    :return:2d array containing arrays of audio samples
    """
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

def get_fft(audio_samples, sample_rate, timestamps, key, features_file, push_window=50, num_bins=None):
    """
    Outputs Fourier Transform of raw audio samples to `featuress` directory

    :param audio_samples: raw audio samples in time domain
    :param sample_rate: sample rate of raw audio
    :param timestamps: timestamps associated with each audio sample in `audio_samples`
    :param key: the key whose fft is being computed
    :param features_file: file to contain the output of the Fourier Transform
    :param push_window: Time length in ms of the input time domain signal to compute FFT of
    :param num_bins: Number of frequency bins from fft computation we should output
    """

    for i in range(audio_samples.shape[0]):
        plt.title("key =  " + key)
        t = np.arange(0, audio_samples.shape[1] / sample_rate, 1.0 / sample_rate)

        # We define the touch peak as the first peak over height 500 in the time series
        peaks, props = find_peaks(audio_samples[i], height=500, distance=sample_rate/500)
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

        # The push peak region is defined such that the touch peak is at 1/8 of the push region length
        start = peaks[0] - int(sample_rate / 1000 * push_window / 8)
        end = peaks[0] + int(sample_rate / 1000 * 7 * push_window / 8)

        touch_samples = audio_samples[i][start : end]
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

        """
        if key == "q":
            #plt.plot(touch_t, touch_samples)
            #plt.xlabel("Frequency (Hz)")
            #plt.ylabel("Amplitude")
            #plt.plot(freq, magnitude)
            plt.plot(t, res[i])
            plt.show()
        """

def get_touch_time_series(res, sample_rate, timestamps, key, features_file, push_window=None, num_bins=None, num_samples=None):
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
        start = peaks[0] - int(sample_rate / 1000 * push_window / 8)
        end = peaks[0] + int(sample_rate / 1000 * 7 * push_window / 8)

        touch_samples = res[i][start : end]
        #touch_samples = np.hanning(len(touch_samples)) * touch_samples
        touch_t = t[start : end]

        if len(touch_samples) < num_samples - 1:
            print("touch samples is ", len(touch_samples))
            print("skipping...")
            continue

        if i == 0:
            np.save(os.path.join(DIR_OUT, KEYBOARD_TYPE, "time_samples_metadata.npy"), touch_t)
        features_file.write(key)
        for sample in touch_samples:
            features_file.write("," + str(sample))
        features_file.write("\n")

        """
        if key == "a":
            #plt.plot(touch_t, touch_samples)
            #plt.plot(freq, magnitude)
            plt.plot(t, res[i])
            plt.show()
        """

if __name__ == "__main__":

    keys = ["space", "backspace", "a", "d", "f", "s", "e", "q", "w", "r", "g"]

    # push peak time window length in milliseconds
    if KEYBOARD_TYPE == "HP_Spectre":
        push_window = 50
    elif KEYBOARD_TYPE == "Dell":
        push_window = 50
    else:
        push_window = 50

    max_freq = 6000
    fs = 44100

    # Number of frequency bins we store
    num_bins = int(push_window * fs / 1000 * max_freq / fs)
    num_samples = int(push_window * fs / 1000)
    print("num_bins ", num_bins)
    print("num_samples ", num_samples)

    features_file = open(os.path.join(DIR_OUT, KEYBOARD_TYPE, "push_fft.csv"), 'w')
    features_file.write("key")

    for i in range(num_bins):
        features_file.write(",freq_bin_" + str(i + 1))
    features_file.write("\n")

    """
    for i in range(num_samples - 1):
        features_file.write(",time_sample_" + str(i + 1))
    features_file.write("\n")
    """

    for key in keys:
        audio_samples, sample_rate, timestamps = get_time_series(key=key, num_samples=35000, entire=False)
        if sample_rate != fs:
            raise ValueError("sample_rate != " + fs)

        get_fft(audio_samples, sample_rate, timestamps, key, features_file, push_window=push_window, num_bins=num_bins)

    features_file.close()
