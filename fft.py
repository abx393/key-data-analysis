import os
import numpy as np

from scipy.io import wavfile
from matplotlib import pyplot as plt
from json import JSONDecoder

DIR = "raw_data"
offset = 10000

for f in os.listdir(DIR):
    print(f)
    (basename, extension) = f.split(".")

    # If it's a wav file, generate fft plot
    if extension == "wav":
        sample_rate, samples = wavfile.read(os.path.join(DIR, f))

        # Get corresponding ground truth JSON file
        labels_file = open(os.path.join(DIR, basename + ".json"))
        labels = JSONDecoder().decode(labels_file.read())
        for timestamp in labels:
            # the key that was pressed
            label = labels[timestamp]

            # timestamp is milliseconds since start of audio
            timestamp = int(timestamp)
            print("timestamp ", timestamp)

            # Get the range of samples associated with the current key press
            sample_start = timestamp * sample_rate // 1000 - offset
            sample_end = timestamp * sample_rate // 1000 + offset
            print("sample_start ", sample_start)
            print("sample_end ", sample_end)

            # number of time samples or number of frequency bins
            n = sample_end - sample_start

            freq = np.fft.fftfreq(n, 1 / sample_rate)
            magnitude = np.fft.fft(samples[sample_start : sample_end])

            # only look at positive frequencies
            freq = freq[: n//2]
            magnitude = magnitude[: n//2]

            plt.plot(freq, magnitude)
            # plt.axis([0, 10000, 0, 300000])
            plt.xlim([0, 5000])
            plt.ylim(bottom=0)
            plt.title("key = {}, time = {} ms".format(label, timestamp))
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.show()

        labels_file.close()

