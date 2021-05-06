"""
Produces spectrogram plots.
"""

import os
import pandas as pd

from scipy.io import wavfile
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
from json import JSONDecoder

DIR_IN = "native_raw_data"
DIR_OUT = "plots/spectrograms"
KEYBOARD_TYPE = "membrane"

for f in os.listdir(os.path.join(DIR_IN, KEYBOARD_TYPE)):
    (basename, extension) = f.split(".")

    # If it's a wav file, process it
    if extension == "wav":
        sample_rate, samples = wavfile.read(os.path.join(DIR_IN, KEYBOARD_TYPE, f))
        samples = samples[:,1]

        """
        # Get corresponding ground truth JSON file
        labels_file = open(os.path.join(DIR_IN, KEYBOARD_TYPE, basename + ".json"))
        labels = JSONDecoder().decode(labels_file.read())
        """

        # Get corresponding ground truth CSV file
        labels_file = open(os.path.join(DIR_IN, KEYBOARD_TYPE, basename + ".csv"))
        df = pd.read_csv(labels_file)
        labels_file.close()

        # Compute spectrogram
        freqs, times, spec = spectrogram(samples, sample_rate, nperseg=10000)

        # Plot spectrogram using color map
        plt.pcolormesh(times, freqs, spec, vmin=0, vmax=100)
        plt.title("timestamp = {}, Keyboard = {}".format(basename, KEYBOARD_TYPE))
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim([0, 3000])

        # Annotate spectogram plot with key labels
        ax = plt.gca()
        for i in range(len(df)):
            label = df.iloc[i, 0]
            timestamp = df.iloc[i, 1]
            print("key ", label)
            print("timestamp ", timestamp)
            ax.text(timestamp, 2300, label, color="white")

        """
        # deprecated: JSON parsing
        for timestamp in labels:
            label = labels[timestamp]
            timestamp = int(timestamp)
            ax.text(timestamp / 1000, 2300, label, color="white")
        """

        plt.savefig(os.path.join(DIR_OUT, basename + ".jpg"))
        plt.show()
