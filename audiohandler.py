import pyaudio
from tqdm import tqdm


class Listener:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=1000, listen_seconds=1):
        self.pformat = pformat
        self.channels = channels
        self.rate = rate
        self.chunk_length = chunk_length

        self.audio = pyaudio.PyAudio()
        self.buffer = [b'\x00' * chunk_length] * int(rate / chunk_length) * listen_seconds * 2

    def __del__(self):
        self.audio.terminate()
        print("ListenThread terminated")

    def __callback(self, in_data, frame_count, time_info, status_flags):
        self.buffer.pop(0)
        self.buffer.append(in_data)
        return None, pyaudio.paContinue

    def listen(self):
        self.stream = self.audio.open(format=self.pformat,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk_length,
                                      stream_callback=self.__callback)
        self.stream.start_stream()

    def terminate(self):
        self.stream.stop_stream()
        self.stream.close()


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


class Player:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000):
        self.pformat = pformat
        self.channels = channels
        self.rate = rate

        self.audio = pyaudio.PyAudio()
        self.playlist = []

    def __del__(self):
        self.audio.terminate()
        print("Player terminated")

    def play(self, data):
        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True)
        stream.write(data)
        stream.stop_stream()
        stream.close()

    def play_list(self):
        while self.playlist:
            self.play(b''.join(self.playlist.pop(0)))


if __name__ == '__main__':
    listener = Listener()
    listener.listen()
    listener.terminate()
