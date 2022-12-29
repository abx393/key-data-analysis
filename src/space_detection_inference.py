import os
from scipy.io import wavfile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from joblib import load

INPUT_DIR = '../native_raw_data_time_series_phrases'
KEYBOARD = 'HP_Spectre'

OUTPUT_DIR = '../features'

TS_PLOTTING = True

push_window = 0.05

# Not space or space
classes = ['NS', 'S']

clf = load('../models/FFT_Classification_space_{}.bin'.format(KEYBOARD))
scaler = load('../models/std_scaler_space_{}.bin'.format(KEYBOARD))

def main():
    push_time_samples = []
    key_labels = []

    for cnt, f in enumerate(os.listdir(os.path.join(INPUT_DIR, KEYBOARD))):
        print(f)
        cnt += 1
        """
        if cnt < 225:
            continue
        """
        if cnt % 10 == 0:
            print('Progress {} files'.format(cnt))

        if f[-3:] != 'wav':
            continue
        sample_rate, samples = wavfile.read(os.path.join(INPUT_DIR, KEYBOARD, f))

        # metadata file
        f_md = open(os.path.join(INPUT_DIR, KEYBOARD, f[:-3] + 'csv'))
        df = pd.read_csv(f_md)
        f_md.close()

        peaks, props = find_peaks(samples, height=600, distance=0.075 * sample_rate)
        ground_truth = {}

        t = np.arange(len(samples)) / sample_rate

        plt.figure(figsize=(20, 5))
        plt.plot(samples)
        ax = plt.gca()
        ax.text(100, 2600, "Predictions", fontsize=10)
        ax.text(100, -2600, "Ground Truth", fontsize=10)

        ground_truth_time_shift = peaks[0] - df.iloc[0, 1] * sample_rate
        print('ground_truth_time_shift', ground_truth_time_shift)

        for j in range(len(df)):
            #ax.axvline(x=df.iloc[j, 1] * sample_rate, linestyle='dashed', alpha=0.5)
            #ax.axvline(x=df.iloc[j, 1] * sample_rate + ground_truth_time_shift, color='red', linestyle='dashed', alpha=0.5)
            #ax.text(x=df.iloc[j, 1] * sample_rate + ground_truth_time_shift, y=-2600, s=df.iloc[j, 0])

            ground_truth[df.iloc[j, 1]] = df.iloc[j, 0]

        for peak in peaks:
            color = 'black'
            push_samples = samples[int(peak - sample_rate * push_window / 8) : int(peak + sample_rate * 7 * push_window / 8)]
            push_time_samples.append(push_samples)
            freq, magnitude = fft(push_samples)

            x = [magnitude]
            x = scaler.transform(x)

            pred = clf.predict(x)
            pred = classes[int(pred)]

            #ax.text(peak, 2600, str(pred), fontsize=10)
            rect = mpatches.Rectangle((peak - sample_rate * push_window / 8, -2400),
                                      push_window * sample_rate, 4800, fill=False, color=color)
            ax.add_patch(rect)

        plt.title('Spacebar Detection Inference Model')
        plt.xlabel('Number of Audio Samples')
        plt.ylim([-3000, 3000])

        if TS_PLOTTING:
            plt.show()

        # Display 1 second worth of data at a time
        """
        for i in range(int(t[-1]) + 1):
            print('i=', i)
            t_sub = t[i * sample_rate : (i+1) * sample_rate]
            samples_sub = samples[i * sample_rate : (i+1) * sample_rate]

            if TS_PLOTTING:
                plt.figure(figsize=(20, 5))
                plt.plot(t_sub, samples_sub)
                ax = plt.gca()

            for j in range(len(df)):
                if df.iloc[j, 1] > i and df.iloc[j, 1] < i+1:
                    if TS_PLOTTING:
                        ax.axvline(x=df.iloc[j, 1], linestyle='dashed', alpha=0.5)
                        ax.text(x=df.iloc[j, 1], y=1000, s=df.iloc[j, 0])

                    ground_truth[df.iloc[j, 1]] = df.iloc[j, 0]

            for peak in peaks:
                if t[peak] > i and t[peak] < i+1:
                    color = 'black'
                    push_samples = samples[int(peak - sample_rate * push_window / 8) : int(peak + sample_rate * 7 * push_window / 8)]
                    push_time_samples.append(push_samples)
                    freq, magnitude = fft(push_samples)

                    x = [magnitude]
                    x = scaler.transform(x)

                    pred = clf.predict(x)
                    pred = classes[int(pred)]

                    if TS_PLOTTING:
                        ax.text(t[peak], 2500, 'pred={}'.format(pred), fontsize=10)
                        rect = mpatches.Rectangle((t[peak] - push_window / 8, -1500),
                                                  push_window, 3000, fill=False, color=color)
                        ax.add_patch(rect)

            plt.title('Spacebar Detection Inference Model')
            plt.xlabel('Time (s)')
            plt.ylim([-3000, 3000])
            if TS_PLOTTING:
                plt.show()
        """
        #print("len(ground_truth) ", len(ground_truth))
        #print("len(peaks) ", len(peaks))

    sample_rate = 44100

    push_time_samples = np.array(push_time_samples)
    print(np.shape(push_time_samples))
    space_cnt = 0
    for label in key_labels:
        if label == 'space':
            space_cnt+= 1
    print('num(space) ', space_cnt)
    print('num(non-space) ', len(key_labels) - space_cnt)

def fft(samples, push_window=push_window, sample_rate=44100, max_freq=6000):
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
        print('freq len is ', len(freq))
        print('skipping...')
        exit(-1)
    else:
        return freq, magnitude

if __name__ == '__main__':
    main()
