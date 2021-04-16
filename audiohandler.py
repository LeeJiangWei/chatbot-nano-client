import threading

import pyaudio
from tqdm import tqdm


class ListenThread(threading.Thread):
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=1000, listen_seconds=1):
        super(ListenThread, self).__init__(daemon=True)
        self.running = False

        self.pformat = pformat
        self.channels = channels
        self.rate = rate
        self.chunk_length = chunk_length

        self.audio = pyaudio.PyAudio()
        self.lock = threading.Lock()
        self.buffer = [b'\x00' * chunk_length] * int(rate / chunk_length) * listen_seconds * 2

    def __del__(self):
        self.audio.terminate()
        print("ListenThread terminated")

    def run(self):
        self.running = True

        print("Start listening...")
        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=self.chunk_length)
        while self.running:
            chunk = stream.read(self.chunk_length)
            self.lock.acquire()
            self.buffer.pop(0)
            self.buffer.append(chunk)
            self.lock.release()

        stream.stop_stream()
        stream.close()
        print("Stop listening...")

    def terminate(self):
        self.running = False


class Recorder:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=1000):
        self.pformat = pformat
        self.channels = channels
        self.rate = rate
        self.chunk_length = chunk_length

        self.audio = pyaudio.PyAudio()
        self.buffer = []

    def __del__(self):
        self.audio.terminate()
        print("Recorder terminated")

    def record(self, record_seconds=3):
        print("Start recording...")
        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=self.chunk_length)

        for _ in tqdm(range(record_seconds)):
            chunk = stream.read(self.rate)
            self.buffer.append(chunk)

        stream.stop_stream()
        stream.close()
        print("Stop recording...")

        return self.buffer

    def clear_buffer(self):
        self.buffer = []


class PlayList:
    pass


if __name__ == '__main__':
    recorder = Recorder()
    buffer = recorder.record()

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    output=True)

    stream.write(b''.join(buffer))

    stream.stop_stream()
    stream.close()
