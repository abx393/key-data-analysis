import os
import time
import keyboard
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 8
DIR_OUT = "native_raw_data"
SUBDIR_OUT = "membrane"

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
ground_truth = "key,time\n"

def on_press(event):
    global ground_truth
    ground_truth = ground_truth + "{},{}\n".format(event.name, event.time - time_start)


p = pyaudio.PyAudio()

time_start = time.time()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

keyboard.on_press(on_press)

print("Recording started...")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording stopped.")

stream.stop_stream()
stream.close()
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
