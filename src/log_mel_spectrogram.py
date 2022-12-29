"""
Produces log mel spectrogram plots.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display

from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

DIR_IN = "../native_raw_data"
DIR_OUT = "../features/spectrogram"
KEYBOARD_TYPE = "HP_Spectre"

cnt = 0
label_cnt = {}
for f in os.listdir(os.path.join(DIR_IN, KEYBOARD_TYPE)):
    (basename, extension) = f.split(".")

    # If it's a wav file, process it
    if extension == "wav":
        sample_rate, samples = wavfile.read(os.path.join(DIR_IN, KEYBOARD_TYPE, f))
        if len(samples.shape) > 1:
            samples = samples[:,1]

        # Get corresponding ground truth CSV file
        labels_file = open(os.path.join(DIR_IN, KEYBOARD_TYPE, basename + ".csv"))
        df = pd.read_csv(labels_file)
        labels_file.close()

        # Annotate spectogram plot with key labels
        num_samples = 44100
        push_window = 75
        ax = plt.gca()

        for i in range(len(df)):
            label = df.iloc[i, 0]
            if label == 'mouse_click':
                continue
            timestamp = df.iloc[i, 1]
            print('timestamp ', timestamp)
            print('key ', label)
            print('timestamp ', timestamp)
            ax.text(timestamp, 2300, label, color='white')

            samples_key = samples[int(sample_rate * timestamp - num_samples / 2) :
                                  int(sample_rate * timestamp + num_samples / 2)]
            samples_key = samples
            peaks, props = find_peaks(samples_key, height=500, distance=sample_rate/500)

            if len(peaks) == 0:
                continue
            if label == '\\':
                continue
            cnt += 1

            if label in label_cnt:
                label_cnt[label] += 1
            else:
                label_cnt[label] = 1

            # The push peak region is defined such that the touch peak is at 1/8 of the push region length
            start = peaks[0] - int(sample_rate / 1000 * push_window / 8)
            end = peaks[0] + int(sample_rate / 1000 * 7 * push_window / 8)

            samples_push_peak = samples_key[start : end]

            mel_spec = librosa.feature.melspectrogram(y=1.0 * samples_push_peak, sr=sample_rate, n_fft=1024, hop_length=512)
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            librosa.display.specshow(mel_spec, y_axis='mel', fmax=8000, x_axis='time')

            np.save(os.path.join(DIR_OUT, KEYBOARD_TYPE, '{}_{}'.format(label, label_cnt[label])), mel_spec)
            #print('mel_spec', mel_spec)

            plt.title('Mel Spectrogram key = {}'.format(label))
            plt.colorbar(format='%+2.0f dB')

            plt.show()
        print('cnt ', cnt)
        print('label_cnt', label_cnt)