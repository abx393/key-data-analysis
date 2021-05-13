"""
Produces Fourier Transform plots.
"""

import os
import numpy as np
import pandas as pd

from scipy.io import wavfile
from matplotlib import pyplot as plt
from json import JSONDecoder

DIR_IN = "native_raw_data"
DIR_OUT = "plots/fft"
KEYBOARD_TYPE = "mechanical"

offset = 10000

for f in os.listdir(os.path.join(DIR_IN, KEYBOARD_TYPE)):
    print(f)
    (basename, extension) = f.split(".")

    # If it's a wav file, generate fft plot
    if extension == "wav":
        sample_rate, samples = wavfile.read(os.path.join(DIR_IN, KEYBOARD_TYPE, f))
        print(samples)
        samples = samples[:, 1]

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

            # Get the range of samples associated with the current key press
            sample_start = int(timestamp * sample_rate - offset)
            sample_end = int(timestamp * sample_rate + offset)

            # number of time samples or number of frequency bins
            n = sample_end - sample_start

            freq = np.fft.fftfreq(n, 1 / sample_rate)
            magnitude = np.fft.fft(samples[sample_start : sample_end])

            # only look at positive frequencies
            freq = freq[: n//2]
            magnitude = magnitude[: n//2]

            plt.plot(freq, magnitude)
            # plt.axis([0, 10000, 0, 300000])
            plt.xlim([0, 3000])
            plt.ylim(bottom=0)
            plt.title("key = {}, time = {} s, keyboard = {}".format(label, timestamp, KEYBOARD_TYPE))
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.savefig(os.path.join(DIR_OUT, basename + "_" + label + ".jpg"))
            plt.show()

        labels_file.close()

