"""
Script to collect audio of key presses, mouse clicks, and mouse scrolls
"""

import os
import time
import keyboard
from pynput.mouse import Listener
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 8
DIR_OUT = "native_raw_data"
SUBDIR_OUT = "Dell"

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
ground_truth = "key,time\n"
last_mouse_scroll = 0

def on_key_press(event):
    global ground_truth
    ground_truth = ground_truth + "{},{}\n".format(event.name, event.time - time_start)

def on_mouse_click(x, y, button, pressed):
    if pressed:
        time_curr = time.time()
        global ground_truth
        ground_truth = ground_truth + "mouse_click,{}\n".format(time_curr - time_start)

def on_mouse_scroll(x, y, dx, dy):
    time_curr = time.time()
    global ground_truth, last_mouse_scroll
    if time_curr - last_mouse_scroll < 1.0:
        return
    ground_truth = ground_truth + "mouse_scroll,{}\n".format(time_curr - time_start)
    last_mouse_scroll = time_curr
    #print("x ", x)
    #print("y ", y)
    #print("dx ", dx)
    #print("dy ", dy)
    #print()

p = pyaudio.PyAudio()

time_start = time.time()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

keyboard.on_press(on_key_press)

listener = Listener(on_click=on_mouse_click, on_scroll=on_mouse_scroll)
listener.start()

print("Recording started...")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording stopped.")

stream.stop_stream()
stream.close()

listener.stop()

print("ground_truth ", ground_truth)
p.terminate()

wf = wave.open(os.path.join(DIR_OUT, SUBDIR_OUT, timestamp + ".wav"), 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

tf = open(os.path.join(DIR_OUT, SUBDIR_OUT, timestamp + ".csv"), 'w')
tf.write(ground_truth)
tf.close()
