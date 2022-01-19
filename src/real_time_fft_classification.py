import os
import time
import numpy as np
import keyboard
import pyaudio
import wave

from joblib import load
from scipy.signal import find_peaks

push_window = 50
max_freq = 6000

buffer = []
keyboard_type = 'HP_Spectre'

clf = load('../models/FFT_Classification_space_{}.bin'.format(keyboard_type))
scaler = load('../models/std_scaler_space_{}.bin'.format(keyboard_type))

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 0.1

MAX_BUFFER_SIZE = 3096
num_bins = int(push_window * RATE / 1000 * max_freq / RATE)
# classes = ['a', 'backspace', 'd', 'e', 'f', 'g', 'q', 'r', 's', 'space', 'w']
classes = ["not space", "space"]


p = pyaudio.PyAudio()
callback_count = 0

def stream_callback(in_data, frame_count, time_info, status):
    global callback_count, buffer
    if callback_count % 3 == 0:
        buffer = []
    callback_count += 1

    # print("frame count ", frame_count)
    for i in range(0, len(in_data) - 2, 2):
        buffer.append(int.from_bytes(in_data[i : i + 2], byteorder='little', signed=True))
        """
        if len(buffer) > MAX_BUFFER_SIZE:
            print("check")
            buffer.pop(0)
        """

    if callback_count % 3 == 0:
        # Find peaks with height at least 500
        peaks, props = find_peaks(buffer, height=500, distance=RATE/500)

        if len(peaks) > 0:
            print("peaks[0] ", peaks[0])

            # The push peak region is defined such that the touch peak is at 1/8 of the push region length
            start = peaks[0] - int(RATE / 1000 * push_window / 8)
            end = peaks[0] + int(RATE / 1000 * 7 * push_window / 8)

            touch_samples = buffer[start : end]
            #touch_samples = np.hanning(len(touch_samples)) * touch_samples
            if len(touch_samples) == 0:
                print("len(touch_samples) ", len(touch_samples))
                print("end - start ", (end - start))
                return (None, pyaudio.paContinue)

            freq = np.fft.rfftfreq(len(touch_samples), 1 / RATE)
            magnitude = np.fft.rfft(touch_samples)

            # only look at positive frequencies
            freq = freq[: len(touch_samples) // 2]
            magnitude = np.abs(magnitude[: len(touch_samples) // 2])

            # Normalize fft
            magnitude /= np.max(magnitude)

            # Remove higher frequencies
            freq = freq[: num_bins]
            magnitude = magnitude[: num_bins]
            if len(freq) < num_bins:
                print("freq len is ", len(freq))
                print("skipping...")
                return (None, pyaudio.paContinue)

            x = [magnitude]
            x = scaler.transform(x)

            prob = clf.predict_proba(x)
            print("predict probabilities: ", prob)
            print("predict key: ", classes[np.argmax(prob)])

    return (None, pyaudio.paContinue)

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                stream_callback=stream_callback)

print("Recording started...")
stream.start_stream()

while stream.is_active():
    time.sleep(RECORD_SECONDS)

print("Recording stopped.")
stream.stop_stream()
stream.close()

p.terminate()

"""
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    print(int.from_bytes(data, byteorder="little", signed=True))
    print(data, "\n")
    frames.append(data)
"""
