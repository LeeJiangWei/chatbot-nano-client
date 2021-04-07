import os
from datetime import datetime
from tqdm import tqdm
import numpy as np

import pyaudio
import wave
import librosa

import classifier

classifier.load_graph("./models/CRNN/CRNN_L.pb")
labels = classifier.load_labels("./models/labels.txt")

# document see http://people.csail.mit.edu/hubert/pyaudio/docs/

CHUNK_LENGTH = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_LENGTH)

print("* recording")
frames = [stream.read(RATE * RECORD_SECONDS)]
print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

w = b''.join(frames)
classifier.run_graph(w, labels, 5)
exit(0)

output_dir = "data/"
output_path = output_dir + "{:%Y%m%d_%H%M%S}.wav".format(datetime.now())

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

wf = wave.open(output_path, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

with open(output_path, 'rb') as f:
    w = f.read()
    classifier.run_graph(w, labels, 5)
