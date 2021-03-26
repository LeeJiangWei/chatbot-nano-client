import os
from datetime import datetime
from tqdm import tqdm

import pyaudio
import wave

# document see http://people.csail.mit.edu/hubert/pyaudio/docs/

CHUNK_LENGTH = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_LENGTH)

print("* recording")

frames = []

for i in tqdm(range(0, int(RATE / CHUNK_LENGTH * RECORD_SECONDS))):
    data = stream.read(CHUNK_LENGTH)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

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
