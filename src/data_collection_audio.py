"""
Script to collect audio of key presses, mouse clicks, and mouse scrolls
"""

import os
import time
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
DIR_OUT = '../test6'
KEYBOARD = 'HP_Spectre'

p = pyaudio.PyAudio()
timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print('Recording started...')
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print('Recording stopped.')

stream.stop_stream()
stream.close()

p.terminate()

wf = wave.open(os.path.join(DIR_OUT, KEYBOARD, timestamp + '.wav'), 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
