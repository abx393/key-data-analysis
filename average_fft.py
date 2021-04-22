"""
Computes the average Fourier Transform for each key.
"""

import os
import numpy as np

from scipy.io import wavfile
from matplotlib import pyplot as plt
from json import JSONDecoder

DIR_IN = "raw_data"
DIR_OUT = "plots/average_fft"
KEYBOARD_TYPE = "mechanical"

offset = 20000

avg_fft = {}
key_count = {}

for f in os.listdir(os.path.join(DIR_IN, KEYBOARD_TYPE)):
    print(f)
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

            # timestamp is milliseconds since start of audio
            timestamp = int(timestamp)

            # Get the range of samples associated with the current key press
            sample_start = timestamp * sample_rate // 1000 - offset
            sample_end = timestamp * sample_rate // 1000 + offset

            # number of time samples or number of frequency bins
            n = sample_end - sample_start

            freq = np.fft.fftfreq(n, 1 / sample_rate)
            ampltitude = np.fft.fft(samples[sample_start : sample_end])

            # only look at positive frequencies
            freq = freq[: n//2]
            ampltitude = ampltitude[: n//2]

            # Store fft's by key
            if label not in avg_fft:
                avg_fft[label] = ampltitude
                key_count[label] = 1
            else:
                print("before avg_fft[label] ", avg_fft[label])
                avg_fft[label] += ampltitude
                print("after avg_fft[label] ", avg_fft[label])
                key_count[label] += 1

            """
            plt.plot(freq, ampltitude)
            # plt.axis([0, 10000, 0, 300000])
            plt.xlim([0, 5000])
            plt.ylim(bottom=0)
            plt.title("key = {}, time = {} ms".format(label, timestamp))
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.show()
            """

        labels_file.close()

print("key_count " + str(key_count))
# print("avg_fft " + average_fft)

fs = 48000
for key in avg_fft:
    # print("key ", key)
    amplitude = avg_fft[key]
    # print("amplitude ", amplitude)

    # Normalize by total number of data points for this key
    # ampltitude /= key_count[key]

    freq = np.fft.fftfreq(offset * 2, 1 / fs)[: offset]

    # Plot the average fft for this key
    plt.clf()
    plt.plot(freq, ampltitude)
    plt.xlim([0, 5000])
    plt.ylim(bottom=0)
    plt.title("Average FFT for key = {}".format(key))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.savefig(os.path.join(DIR_OUT, key + ".jpg"))
    plt.show()


