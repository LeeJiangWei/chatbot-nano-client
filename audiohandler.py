import wave
import audioop
import pyaudio
from tqdm import tqdm
import webrtcvad
import time
import multiprocessing

CHUNK_LENGTH=1024

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


            if exit_count > 400 or stop_count > 30:
                break

            # if len(self.buffer) > buffer_window_len and rms < silence_threshold:
            #     break

        stream.stop_stream()

        stream.close()

        return self.buffer, exit_count < 100


class Player:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=8000):
        # should match TTS module
        self.pformat = pformat
        self.channels = channels
        self.rate = rate

        self.audio = pyaudio.PyAudio()

        self.wav_data=None
        self.wakeup_event=None
        self.seek=0

    def __del__(self):
        self.audio.terminate()

    def play(self, data):
        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True)
        # interrupt
        stream.write(data)
        # to solve the suddenly cut off of audio
        time.sleep(stream.get_output_latency()*2)
        stream.stop_stream()
        stream.close()

    def _callback(self, in_data, frame_count, time_info, status):
        start = self.seek
        self.seek += frame_count * pyaudio.get_sample_size(self.pformat)
        if self.seek < len(self.wav_data):
            print("call")
            return self.wav_data[start:self.seek], pyaudio.paContinue
        else:
            return self.wav_data[start:], pyaudio.paContinue

    def play_unblock(self, data, wakeup_event):
        self.wav_data = data
        self.wakeup_event = wakeup_event
        self.seek = 0

        stream = self.audio.open(format = self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True,
                                 frames_per_buffer=CHUNK_LENGTH,
                                 stream_callback=self._callback)
        stream.start_stream()
        while stream.is_active():
            # 在播放音频的时候，每0.1s检测一次是否被唤醒，如果被唤醒则停止播音
            if wakeup_event.is_set():
                break
            time.sleep(0.1)
        time.sleep(stream.get_output_latency()*4)
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

    # def play_wav2(self, path):
    #     with wave.open(path) as wf:
    #         stream = self.audio.open(format=
    #                                  self.audio.get_format_from_width(wf.getsampwidth()),
    #                                  channels=wf.getnchannels(),
    #                                  rate=wf.getframerate(),
    #                                  output=True,
    #                                  stream_callback=self.__callback)
    #         stream.write(wf.readframes(wf.getnframes()))
    #         stream.stop_stream()
    #         stream.close()
    #
    # def __callback(self, in_data, frame_count, time_info, status_flags):
    #
    #     return None, pyaudio.paContinue


if __name__ == '__main__':
    player=Player(rate=16000)
    with wave.open("./juice2.wav",'rb') as f:
        wave_data= f.readframes(f.getnframes())
    value = multiprocessing.Event()
    player.play_unblock(wave_data, value)


