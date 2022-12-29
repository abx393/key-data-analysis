"""
Script to collect audio of key presses, mouse clicks, and mouse scrolls
"""

import os
import time
import keyboard
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
DIR_OUT = '../test6'
KEYBOARD = 'HP_Spectre'

timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
ground_truth = 'key,time\n'
last_mouse_scroll = 0

def on_key_press(event):
    global ground_truth
    global stream
    #print('event.time ', event.time)
    #print('stream.get_time() ', stream.get_time())
    ground_truth += '{},{}\n'.format(event.name, event.time - time_start)
    #ground_truth = ground_truth + '{},{}\n'.format(event.name, stream.get_time() - time_start)

time_start = time.time()
print('time_start', time_start)

while True:
    keyboard.on_press(on_key_press)

print('ground_truth ', ground_truth)
tf = open(os.path.join(DIR_OUT, KEYBOARD, timestamp + '.csv'), 'w')
tf.write(ground_truth)
tf.close()
