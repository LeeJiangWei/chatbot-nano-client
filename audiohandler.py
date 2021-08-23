import wave
import audioop
import pyaudio
from tqdm import tqdm
import webrtcvad


class Listener:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=1000, listen_seconds=1):
        # should match KWS module
        self.pformat = pformat
        self.channels = channels
        self.rate = rate
        self.chunk_length = chunk_length
        self.listen_seconds = listen_seconds

        self.audio = pyaudio.PyAudio()
        self.buffer = [b'\x01' * chunk_length * pyaudio.get_sample_size(pformat)] * int(
            rate / chunk_length) * listen_seconds * 2

    def __del__(self):
        self.audio.terminate()

    def __callback(self, in_data, frame_count, time_info, status_flags):
        self.buffer.pop(0)
        self.buffer.append(in_data)
        return None, pyaudio.paContinue

    def listen(self):
        self.buffer = [b'\x01' * self.chunk_length * pyaudio.get_sample_size(self.pformat)] * int(
            self.rate / self.chunk_length) * self.listen_seconds * 2

        self.stream = self.audio.open(format=self.pformat,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk_length,
                                      stream_callback=self.__callback)
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()


class Recorder:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=1000):
        # should match ASR module
        self.pformat = pformat
        self.channels = channels
        self.rate = rate
        self.chunk_length = chunk_length

        self.audio = pyaudio.PyAudio()
        self.buffer = []

    def __del__(self):
        self.audio.terminate()

    def record(self, record_seconds=3):
        self.buffer = []
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

        return self.buffer

    def record_auto(self, silence_threshold=200, max_silence_second=1):
        self.buffer = []
        # width = pyaudio.get_sample_size(self.pformat)
        # buffer_window_len = int(self.rate / self.chunk_length * max_silence_second)

        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=self.chunk_length)

        vad = webrtcvad.Vad(mode=2)
        start_count = 0
        stop_count = 0
        exit_count = 0
        while True:
            chunk = stream.read(int(self.chunk_length * self.rate / 1000))
            # self.buffer.append(chunk)

            # data = b''.join(self.buffer[-buffer_window_len:])
            # rms = audioop.rms(data, width)
            # print("%s\r" % rms)

            vad_flags = vad.is_speech(chunk, sample_rate=self.rate)

            # ready
            if start_count < 10:
                if vad_flags:
                    start_count += 1
                    # print("listening")
                    self.buffer.append(chunk)
                    exit_count = max(0, exit_count - 1)
                else:
                    self.buffer = []
                    # print('sleeping..')
                    start_count = 0
                    exit_count += 1
            # start
            else:
                # print('recording...')
                self.buffer.append(chunk)
                exit_count = 0
                if not vad_flags:
                    stop_count += 1
                    exit_count += 1

            if exit_count > 100 or stop_count > 15:
                break

            # if len(self.buffer) > buffer_window_len and rms < silence_threshold:
            #     break

        stream.stop_stream()

        stream.close()

        return self.buffer, exit_count < 100


class Player:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=22050):
        # should match TTS module
        self.pformat = pformat
        self.channels = channels
        self.rate = rate

        self.audio = pyaudio.PyAudio()

    def __del__(self):
        self.audio.terminate()

    def play(self, data):
        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True)
        stream.write(data)
        stream.stop_stream()
        stream.close()

    def play_wav(self, path):
        with wave.open(path) as wf:
            stream = self.audio.open(format=
                                     self.audio.get_format_from_width(wf.getsampwidth()),
                                     channels=wf.getnchannels(),
                                     rate=wf.getframerate(),
                                     output=True)
            stream.write(wf.readframes(wf.getnframes()))
            stream.stop_stream()
            stream.close()


if __name__ == '__main__':
    recorder = Recorder()
    buffer = recorder.record_auto()
