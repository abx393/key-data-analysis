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

DIR_IN = "../native_raw_data_time_series_phrases"
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
        ax = plt.gca()
        for j in range(len(df)):
            print('df.iloc[j, 1] ', df.iloc[j, 1])
            ax.axvline(x=df.iloc[j, 1], linestyle='dashed', alpha=0.5)
            #ax.axvline(x=df.iloc[j, 1] * sample_rate + ground_truth_time_shift, color='red', linestyle='dashed', alpha=0.5)
            ax.text(x=df.iloc[j, 1], y=5000, s=df.iloc[j, 0], color='white')
            #ax.text(x=df.iloc[j, 1] * sample_rate + ground_truth_time_shift, y=-2600, s=df.iloc[j, 0])
            #ground_truth[df.iloc[j, 1]] = df.iloc[j, 0]

        mel_spec = librosa.feature.melspectrogram(y=1.0 * samples, sr=sample_rate, n_fft=2048, hop_length=1024)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec, y_axis='mel', fmax=8000, x_axis='time')

        #np.save(os.path.join(DIR_OUT, KEYBOARD_TYPE, '{}_{}'.format(label, label_cnt[label])), mel_spec)
        #print('mel_spec', mel_spec)

        plt.title('Mel Spectrogram')
        plt.colorbar(format='%+2.0f dB')

        plt.show()
        print('cnt ', cnt)
        print('label_cnt', label_cnt)

