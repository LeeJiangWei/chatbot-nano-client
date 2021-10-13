import wave
import time
import multiprocessing

import audioop
import pyaudio
from tqdm import tqdm
import webrtcvad

CHUNK_LENGTH = 1024
RECORDER_CHUNK_LENGTH = 30  # 一个块=30ms的语音
LISTENER_CHUNK_LENGTH = 1000  # 一个块=1s的语音

class Listener:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=LISTENER_CHUNK_LENGTH, listen_seconds=1):
        # should match KWS module
        self.pformat = pformat
        self.channels = channels
        self.rate = rate
        self.chunk_length = chunk_length
        self.listen_seconds = listen_seconds

        self.audio = pyaudio.PyAudio()
        self.buffer = [b"\x01" * chunk_length * pyaudio.get_sample_size(pformat) * self.channels] * int(
            rate / chunk_length) * listen_seconds * 2

    def __del__(self):
        self.audio.terminate()

    def __callback(self, in_data, frame_count, time_info, status_flags):
        self.buffer.pop(0)
        self.buffer.append(in_data)
        return None, pyaudio.paContinue

    def listen(self):
        # 维护一个缓存队列，每个元素是一个块chunk的音频数据，一共有n个，加起来刚好是listen_seconds秒的音频数据
        # 每次回调函数都进行一次出列一次入列，队列中始终包含最近listen_seconds秒的音频数据
        self.buffer = [b"\x01" * self.chunk_length * pyaudio.get_sample_size(self.pformat) * self.channels] * int(
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
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=RECORDER_CHUNK_LENGTH):
        r"""
        Args:
            pformat: 音频样本点的数据类型
            channels (int): 声道数
            rate (int): 采样率，Hz
            chunk_length (int): 一个块对应的时间长度(ms)
        块长度取30ms是因为vad只支持帧长10ms，20ms和30ms
        """
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
        r"""根据给定的时间长度，录制一段音频
        Args:
            record_seconds (int): record time (second)
        """
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
        r"""根据外部环境的声音强度，智能录制有声音的片段。
        连续收到start_thr个有声块(chunk)就认为有人声，开始正式录音（这start_thr个块的数据也会保留）；
        开始录音后连续收到stop_thr个无声块就认为录音结束，退出录音；
        在一次录音过程中累计收到超过exit_thr个无声块时认为本次录音全程静音，退出录音。
        Args:
            silence_threshold (int): deprecated
            max_silence_second (int): deprecated
        Return:
            wav_list (list): 每个元素是一个块chunk的音频数据，wav_list[-1]是最近一个时刻录制的
            no_sound (bool): True表示本次录制没有录到声音，False表示录到了声音
        """
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
        start_thr = 10  # 300ms
        stop_thr = 50  # 1.5s
        exit_thr = 333  # 10s
        while True:
            chunk = stream.read(int(self.chunk_length * self.rate / 1000))  # 除以1000，把ms变成s

            # data = b''.join(self.buffer[-buffer_window_len:])
            # rms = audioop.rms(data, width)
            # print("%s\r" % rms)

            vad_flags = vad.is_speech(chunk, sample_rate=self.rate)

            # ready
            if start_count < start_thr:
                if vad_flags:
                    start_count += 1
                    self.buffer.append(chunk)
                    exit_count = max(0, exit_count - 1)
                else:
                    self.buffer = []
                    start_count = 0
                    exit_count += 1
            # start
            else:
                self.buffer.append(chunk)
                exit_count = 0
                if not vad_flags:
                    stop_count += 1

            if exit_count > exit_thr or stop_count > stop_thr:
                break

            # if len(self.buffer) > buffer_window_len and rms < silence_threshold:
            #     break

        stream.stop_stream()

        stream.close()

        return self.buffer, exit_count >= exit_thr


class Player:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=8000):
        # voiceai的TTS模块目前只支持8000采样率，在播放音频时需要注意保持采样率一致
        self.pformat = pformat
        self.channels = channels
        self.rate = rate

        self.audio = pyaudio.PyAudio()

        self.wav_data = None
        self.wakeup_event = None
        self.seek = 0

        self.play_frames = 0  # 仅用于配合tqdm_iterator使用，表示stream的while播放循环中已经播放了几帧数据

    def __del__(self):
        self.audio.terminate()

    def play(self, wav_data):
        r"""播放二进制音频数据
        Args:
            wav_data (bytes): audio data
        """
        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True)
        # interrupt
        stream.write(wav_data)
        # to solve the suddenly cut off of audio
        time.sleep(stream.get_output_latency())
        stream.stop_stream()
        stream.close()

    def _callback(self, in_data, frame_count, time_info, status):
        start = self.seek
        chunk_length = frame_count * pyaudio.get_sample_size(self.pformat) * self.channels
        self.seek += chunk_length

        data = self.wav_data[start: self.seek]
        self.play_frames += 1

        return data, pyaudio.paContinue

    def play_unblock(self, wav_data, wakeup_event):
        r"""非阻塞地播放一个音频流，期间允许被打断
        Args:
            wav_data (bytes): 音频流二进制数据
            wakeup_event (multiprocessing.Event()): 唤醒事件，用于打断音频播放
        """
        self.wav_data = wav_data
        self.wakeup_event = wakeup_event
        self.seek = 0  # 文件指针

        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True,
                                 frames_per_buffer=CHUNK_LENGTH,
                                 stream_callback=self._callback)
        stream.start_stream()

        chunk_bytes = CHUNK_LENGTH * self.channels * pyaudio.get_sample_size(self.pformat)  # 一个CHUNK所包含的字节数
        # 回调函数返回不完整的块chunk时，stream将不播放该块并结束回调的死循环，因此在播放之前就把wav_data补足到块长度的整数倍
        # 补足chunk_length个字节，第二个取余的目的是如果wav_data本来就是整数倍，就不需要额外加一个静音块
        self.wav_data += b"\x00" * ((chunk_bytes - len(self.wav_data) % chunk_bytes) % chunk_bytes)
        
        # tqdm_iterator仅用于展示播音过程，不想看可以去掉
        # NOTE: 目前tqdm_iterator还有未知问题，有时候正常有时候会在next(tqdm_iterator)处抛出StopIteration异常，只好先不打开
        # n_chunks = len(self.wav_data) // chunk_bytes
        # tqdm_iterator = iter(tqdm(range(n_chunks), "播音中"))

        # self.play_frames, prev = 0, 0
        while stream.is_active():
            # 在播放音频的时候，每0.1s检测一次是否被唤醒，如果被唤醒则停止播音
            if wakeup_event.is_set():
                break
            time.sleep(0.1)
            # for _ in (range(self.play_frames - prev)):
            #     next(tqdm_iterator)
            # prev = self.play_frames
        # 把最后剩下的一点点走完，这里因为是自己iter(tqdm())可能没用对，由于某种原因会剩下一点点
        # for _ in tqdm_iterator:
        #     pass

        time.sleep(stream.get_output_latency())
        # tx2开发板get_output_latency是0.256s
        stream.stop_stream()
        stream.close()

    def play_wav(self, path):
        r"""打开一个音频文件，从头到尾播放完
        Args:
            path (str): path of audio file
        """
        with wave.open(path) as wf:
            stream = self.audio.open(format=
                                     self.audio.get_format_from_width(wf.getsampwidth()),
                                     channels=wf.getnchannels(),
                                     rate=wf.getframerate(),
                                     output=True)
            stream.write(wf.readframes(wf.getnframes()))
            stream.stop_stream()
            stream.close()


if __name__ == "__main__":
    player = Player(rate=16000)
    with wave.open("./juice2.wav", "rb") as wf:
        wav_data = wf.readframes(wf.getnframes())
    event = multiprocessing.Event()
    player.play_unblock(wav_data, event)
