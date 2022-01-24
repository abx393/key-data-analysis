import os
from scipy.io import wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

INPUT_DIR = "../native_raw_data_time_series"
KEYBOARD = "HP_Spectre"

OUTPUT_DIR = "../features"

TS_PLOTTING = True
FFT_PLOTTING = False

push_window = 0.05
push_time_samples = []
key_labels = []

for cnt, f in enumerate(os.listdir(os.path.join(INPUT_DIR, KEYBOARD))):
    cnt += 1
    if cnt >= 225:
        break
    if cnt % 10 == 0:
        print("Progress {} files".format(cnt))

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
    for i in range(int(t[-1]) - 1):
        t_sub = t[i * sample_rate : (i+1) * sample_rate]
        samples_sub = samples[i * sample_rate : (i+1) * sample_rate]

        if TS_PLOTTING:
            plt.figure(figsize=(20, 5))
            plt.title("Push Peak Segmentation")
            plt.plot(t_sub, samples_sub)
            ax = plt.gca()

        for j in range(len(df)):
            if df.iloc[j, 1] > i and df.iloc[j, 1] < i+1:
                if TS_PLOTTING:
                    ax.axvline(x=df.iloc[j, 1], linestyle='dashed', alpha=0.5)
                    ax.text(x=df.iloc[j, 1], y = 1000, s=df.iloc[j, 0])

                ground_truth[df.iloc[j, 1]] = df.iloc[j, 0]

        for peak in peaks:
            if t[peak] > i and t[peak] < i+1:
                color = "black"
                assoc_key = ""
                for time in ground_truth:
                    if abs(t[peak] - time) < 0.03:
                        color = "red"
                        assoc_key = ground_truth[time]

                if TS_PLOTTING:
                    ax.text(t[peak], 2500, "push ({})".format(assoc_key), fontsize=12)
                    rect = mpatches.Rectangle((t[peak] - push_window / 8, -1500),
                                              push_window, 3000, fill=False, color=color)
                    ax.add_patch(rect)

                if assoc_key != "":
                    push_samples = samples[int(peak - sample_rate * push_window / 8) : int(peak + sample_rate * 7 * push_window / 8)]
                    push_time_samples.append(push_samples)
                    key_labels.append(assoc_key)

        plt.xlabel("Time (s)")
        plt.ylim([-3000, 3000])
        if TS_PLOTTING:
            plt.show()
    #print("len(ground_truth) ", len(ground_truth))
    #print("len(peaks) ", len(peaks))


sample_rate = 44100

push_time_samples = np.array(push_time_samples)
print(np.shape(push_time_samples))
space_cnt = 0
for label in key_labels:
    if label == "space":
        space_cnt+= 1
print("num(space) ", space_cnt)
print("num(non-space) ", len(key_labels) - space_cnt)

features_file = open(os.path.join(OUTPUT_DIR, KEYBOARD, "push_fft_natural_dataset.csv"), "w")

for i, samples in enumerate(push_time_samples):
    max_freq = 6000

    # Number of frequency bins we store
    num_bins = int(push_window * sample_rate * max_freq / sample_rate)
    freq = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    magnitude = np.fft.rfft(samples)

    # only look at positive frequencies
    freq = freq[: len(samples) // 2]
    magnitude = np.abs(magnitude[: len(samples) // 2])

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
        features_file.write("key")
        for j in range(len(freq)):
            features_file.write(",{}".format(freq[j]))
        features_file.write("\n")

    features_file.write(key_labels[i])
    for j in range(len(freq)):
        features_file.write(",{}".format(magnitude[j]))
    features_file.write("\n")

    if FFT_PLOTTING:
        plt.title("Key={}".format(key_labels[i]))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.plot(freq, magnitude)
        plt.show()

features_file.close()
