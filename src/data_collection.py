"""
Script to collect audio of key presses, mouse clicks, and mouse scrolls
"""

import ctypes
import os
import time
import sys
import timeit
import keyboard
from pynput.mouse import Listener
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
DIR_OUT = 'test8'
KEYBOARD = 'HP_Spectre'

timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
ground_truth = 'key,time\n'
last_mouse_scroll = 0

def get_curr_time():
    "return a timestamp in milliseconds (ms)"
    tics = ctypes.c_int64()
    freq = ctypes.c_int64()

    # get ticks on the internal ~2MHz QPC clock
    ctypes.windll.Kernel32.QueryPerformanceCounter(ctypes.byref(tics))
    # get the actual freq. of the internal ~2MHz QPC clock
    ctypes.windll.Kernel32.QueryPerformanceFrequency(ctypes.byref(freq))

    t_ms = tics.value * 1e3 / freq.value
    return t_ms

def on_key_press(event):
    global ground_truth
    global stream
    #print('event.time ', event.time)
    #print('stream.get_time() ', stream.get_time())
    ground_truth += '{},{}\n'.format(event.name, event.time - time_start)
    #ground_truth = ground_truth + '{},{}\n'.format(event.name, stream.get_time() - time_start)

def on_mouse_click(x, y, button, pressed):
    if pressed:
        time_curr = time.time()
        global ground_truth
        #ground_truth = ground_truth + "mouse_click,{}\n".format(time_curr - time_start)

def on_mouse_scroll(x, y, dx, dy):
    time_curr = time.time()
    global ground_truth, last_mouse_scroll
    if time_curr - last_mouse_scroll < 1.0:
        return
    ground_truth += 'mouse_scroll,{}\n'.format(time_curr - time_start)
    last_mouse_scroll = time_curr

p = pyaudio.PyAudio()

#time_start = time.clock() #timeit.default_timer() #time.perf_counter_ns() / 10 ** 9
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

time_start = time.time()
#time_start = get_curr_time()
print('time_start', time_start)

keyboard.on_press(on_key_press)
#listener = Listener(on_click=on_mouse_click, on_scroll=on_mouse_scroll)
#listener.start()

print('Recording started...')
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    # print('Buffer ', buffer)
    frames.append(data)

print('Recording stopped.')

stream.stop_stream()
stream.close()

#listener.stop()

print('ground_truth ', ground_truth)
p.terminate()

wf = wave.open(os.path.join(DIR_OUT, KEYBOARD, timestamp + '.wav'), 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

tf = open(os.path.join(DIR_OUT, KEYBOARD, timestamp + '.csv'), 'w')
tf.write(ground_truth)
tf.close()
