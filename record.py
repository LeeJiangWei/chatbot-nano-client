import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import io

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

container = io.BytesIO()

wf = wave.open(container, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

container.seek(0)
w = container.read()
classifier.run_graph(w, labels, 5)
