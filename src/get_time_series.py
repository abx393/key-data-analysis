"""
Retrieve raw time series data
"""

import os
import numpy as np
import pandas as pd

from scipy.io import wavfile

DIR_IN = "../native_raw_data"
KEYBOARD_TYPE = "mechanical"

def get_time_series(key=None, path=DIR_IN, num_samples=44000):
    res = []
    for f in os.listdir(os.path.join(path, KEYBOARD_TYPE)):
        (basename, extension) = f.split(".")

        # If it's a wav file, generate fft plot
        if extension == "wav":
            sample_rate, samples = wavfile.read(os.path.join(path, KEYBOARD_TYPE, f))
            # print("samples.shape ", samples.shape)
            if len(samples.shape) == 2:
                samples = np.array(samples[:, 1])

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

                # number of time samples or number of frequency bins
                n = sample_end - sample_start

                res.append(samples[sample_start : sample_end])
    return np.array(res)

if __name__ == "__main__":
    # print(get_time_series("space"))
    # print(type(get_time_series("space")))
    # print(type(get_time_series("backspace")))
    res = get_time_series()
    print(res.shape)
    print(type(res))
    print(res)

