import io
import threading
import pyaudio
import wave
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import tensorflow as tf

import classifier

CHUNK_LENGTH = 1000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
TOPK = 1


class Recorder(threading.Thread):
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=1000, record_seconds=1):
        super(Recorder, self).__init__()
        self.pformat = pformat
        self.channels = channels
        self.rate = rate
        self.chunk_length = chunk_length
        self.audio = pyaudio.PyAudio()
        self.lock = threading.Lock()
        self.buffer = [b'\x00' * chunk_length] * int(rate / chunk_length) * record_seconds * 2

    def __del__(self):
        self.audio.terminate()
        print("Recorder terminated")

    def run(self):
        iter_rate = int(self.rate / self.chunk_length)

        print("Start recording...")
        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=self.chunk_length)
        while True:
            try:
                for _ in range(iter_rate):
                    chunk = stream.read(self.chunk_length)
                    self.lock.acquire()
                    self.buffer.pop(0)
                    self.buffer.append(chunk)
                    self.lock.release()
            except KeyboardInterrupt:
                break

        stream.stop_stream()
        stream.close()
        print("Stop recording...")


if __name__ == '__main__':
    classifier.load_graph("./models/CRNN/CRNN_L.pb")
    labels = classifier.load_labels("./models/labels.txt")

    recorder = Recorder(FORMAT, CHANNELS, RATE, CHUNK_LENGTH, RECORD_SECONDS)
    recorder.setDaemon(True)
    recorder.start()

    plt.ion()
    plt.figure()
    plt.title("Signal Wave")

    with tf.Session() as sess:
        while True:
            frames = recorder.buffer[:int(RATE / CHUNK_LENGTH * RECORD_SECONDS)]

            container = io.BytesIO()
            wf = wave.open(container, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            container.seek(0)
            wav_data = container.read()

            softmax_tensor = sess.graph.get_tensor_by_name("labels_softmax:0")
            mfcc_tensor = sess.graph.get_tensor_by_name("Mfcc:0")
            (predictions,), (mfcc,) = sess.run([softmax_tensor, mfcc_tensor], {"wav_data:0": wav_data})

            top_k = predictions.argsort()[-TOPK:][::-1]
            for node_id in top_k:
                human_string = labels[node_id]
                score = predictions[node_id]
                print('%s (score = %.5f)' % (human_string, score))

            signals = np.frombuffer(b''.join(frames), dtype=np.int16)

            plt.subplot(211)
            plt.ylim([-500, 500])
            plt.plot(signals)
            plt.subplot(212)
            plt.plot(mfcc)
            # plt.imshow(mfcc, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
            plt.pause(0.1)
            plt.clf()
