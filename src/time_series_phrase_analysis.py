import os
from scipy.io import wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_DIR = "../native_raw_data_time_series_phrases"
KEYBOARD = "Dell"

OUTPUT_DIR = "../features"

push_time_samples = []
key_labels = []

for cnt, f in enumerate(os.listdir(os.path.join(INPUT_DIR, KEYBOARD))):
    if f[-3:] != "wav":
        continue
    sample_rate, samples = wavfile.read(os.path.join(INPUT_DIR, KEYBOARD, f))

    # metadata file
    f_md = open(os.path.join(INPUT_DIR, KEYBOARD, f[:-3] + "csv"))
    df = pd.read_csv(f_md)
    f_md.close()

    peaks, props = find_peaks(samples, height=500, distance=0.05 * sample_rate)
    ground_truth = {}

    t = np.arange(len(samples)) / sample_rate
    plt.plot(t, samples)
    ax = plt.gca()

    for j in range(len(df)):
        ax.axvline(x=df.iloc[j, 1], linestyle='dashed', alpha=0.5)
        ax.text(x=df.iloc[j, 1], y=1000, s=df.iloc[j, 0])

    plt.show()
