import sys
import os
import numpy as np
import pandas as pd
import constants
from scipy.io import wavfile
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)

DIR_IN = 'native_raw_data'
DIR_OUT = 'features/time_series_words'
KEYBOARD = 'HP_Spectre'
PLOTTING = True

train_set = {}
x_test = []
y_test = []

for key in constants.LETTERS:
    print("key ", key)
    all_instances = []
    for f in os.listdir(os.path.join(DIR_IN, KEYBOARD)):
        (basename, extension) = f.split('.')
        if extension != 'wav':
            continue

        fs, samples = wavfile.read(os.path.join(DIR_IN, KEYBOARD, f))
        if len(np.shape(samples)) == 2:
            samples = samples[:,0]

        '''
        plt.title("entire")
        plt.plot(samples)
        if PLOTTING:
            plt.show()
        '''

        # Get corresponding ground truth CSV file
        labels_file = open(os.path.join(DIR_IN, KEYBOARD, basename + '.csv'))
        df = pd.read_csv(labels_file)
        #print(df.head())

        df.set_index('key', inplace=True)
        if key in df.index:
            key_time_list = list(df.loc[key])
            if key_time_list[0] == 'time':
                continue

            key_time = key_time_list[0]
            mid_sample = int(fs * key_time)
            key_sample_len = int(0.08 * fs)

            samples_key = samples[mid_sample - key_sample_len : mid_sample + key_sample_len]
            plt.title('before alignment: key ' + key)
            plt.plot(samples_key)

            all_instances.append(samples_key)
        labels_file.close()
    if PLOTTING:
        plt.show()


    all_samples = np.zeros((len(all_instances), int(0.08 * 44100 * 2)))
    for j, samples_key in enumerate(all_instances):
        if np.shape(samples_key)[-1] == 2:
            print(samples_key)

        peaks, props = find_peaks(samples_key, height=500)
        samples_key_list = list(samples_key)
        if len(peaks) == 0:
            continue
        for i in range(peaks[0] - 300):
            samples_key_list.pop(0)

        samples_key = np.array(samples_key_list)
        for i in range(len(samples_key)):
            all_samples[j][i] = samples_key[i]
        for i in range(len(samples_key), np.shape(all_samples)[1]):
            all_samples[j][i] = 0

        plt.title('after alignment: key ' + key)
        plt.plot(samples_key)

        #print("peaks ", peaks)

    if PLOTTING:
        plt.show()

    train_samples = all_samples[: int(0.8 * len(all_samples))]
    test_samples = all_samples[int(0.8 * len(all_samples)):]
    for i in range(len(test_samples)):
        x_test.append(test_samples[i][:2000])
        y_test.append(key)

    if len(train_samples) != 0:
        mean_samples_key = np.mean(train_samples, axis=0)
        mean_samples_key = mean_samples_key[: 2000]
        train_set[key] = mean_samples_key

        plt.title("average " + key)
        plt.plot(mean_samples_key)
        if PLOTTING:
            plt.show()

for i in range(len(x_test)):
    curr_samples = x_test[i]
    ground_truth = y_test[i]

    max_cc = 0
    key_pred = ""
    for key in train_set:
        cc = np.correlate(curr_samples, train_set[key])
        print("Cross-correlation ", cc)
        if abs(cc[0]) > max_cc:
            max_cc = abs(cc[0])
            key_pred = key

    print("Ground truth ", ground_truth)
    print("Prediction ", key_pred)

'''
def align(siga, sigb):
    for i in range(-len(sigb) / 2, len(sigb) / 2, 100):
        if i < 0:
            for j in range()
'''
