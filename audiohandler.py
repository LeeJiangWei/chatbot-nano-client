import wave
import time
import threading

import audioop
import pyaudio
from tqdm import tqdm
import webrtcvad

FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 8000
LISTEN_SECONDS = 1

PLAYER_CHUNK_LENGTH = 1024
RECORDER_CHUNK_LENGTH = 480
LISTENER_CHUNK_LENGTH = 1000

from utils.tts_utils import TTSBiaobei


class Listener:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=1000, listen_seconds=1):
        # should match KWS module
        self.pformat = pformat
        self.channels = channels
        self.rate = rate
        self.chunk_length = chunk_length  # 取整方便buffer切块
        self.listen_seconds = listen_seconds

        self.audio = pyaudio.PyAudio()
        self.refresh_buffer()

    def __del__(self):
        self.audio.terminate()

    def __callback(self, in_data, frame_count, time_info, status_flags):
        # 每次回调函数都对buffer进行一次出列一次入列，使队列中始终包含最近2 * listen_seconds秒的音频数据
        self.buffer.pop(0)
        self.buffer.append(in_data)
        return None, pyaudio.paContinue

    def refresh_buffer(self):
        # buffer是一个缓存队列，每个元素是一个块chunk的音频数据，一共有2n个，加起来刚好是2 * listen_seconds秒的音频数据，多出的一半作为冗余
        self.buffer = [b"\x00" * self.chunk_length * pyaudio.get_sample_size(self.pformat) * self.channels] * int(
            self.rate / self.chunk_length) * self.listen_seconds * 2

    def listen(self):
        self.refresh_buffer()
        self.stream = self.audio.open(format=self.pformat,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk_length,
                                      stream_callback=self.__callback)
        self.stream.start_stream()

    def load_wav_data(self, wav_data):
        r"""Author: zhang.haojian
        该函数仅用于调试，可以直接把TTS输出的音频数据放进来，不需要每次都用人声去测，一次可以测很多语句，调试速度更快
        Args:
            wav_data (bytes): 音频数据，wav/pcm皆可
        """
        self.wav_data = wav_data
        self.seek = 0

    def step_a_chunk(self):
        r"""VAD"""
        start = self.seek
        self.seek += int(self.chunk_length * self.channels * pyaudio.get_sample_size(self.pformat))
        new_chunk = self.wav_data[start: self.seek]

        # in_data =
        self.buffer.pop(0)
        self.buffer.append(new_chunk)
        return len(new_chunk)

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()


class Recorder:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=480):
        r"""
        Args:
            pformat: 音频样本点的数据类型
            channels (int): 声道数
            rate (int): 采样率，Hz
            chunk_length (int): 一个块包含的帧frame数
        块长度取480是因为vad只支持帧长10ms，20ms和30ms，采样率16000的情况下30ms对应480帧
        """
        # should match ASR module
        self.pformat = pformat
        self.channels = channels
        self.rate = rate
        self.chunk_length = chunk_length

        self.audio = pyaudio.PyAudio()
        self.buffer = []
        self.no_sound = True  # no_sound写成成员是为了多线程调用record_auto的时候外部依然可以拿到

    def __del__(self):
        self.audio.terminate()

    def record(self, record_seconds=3, finish_record_event=None):
        r"""根据给定的时间长度，录制一段音频，支持流式录制翻译
        finish_record_event不传参即为整段录制后翻译，传Event即可支持流式录制翻译
        Args:
            record_seconds (int): record time (second)
            finish_record_event (threading.Event): 结束录制事件，用来向外部传达信息
        Return:
            buffer (list): 录制的音频，每个元素是一个块chunk，buffer[-1]是最新录制的数据。
                整段录制时才会用到，流式录制不需要访问这个返回值
        """
        if finish_record_event:
            finish_record_event.clear()  # 先复位事件

        self.buffer = []
        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True)

        chunk_time = 0.1  # 0.1s
        for _ in tqdm(range(int(record_seconds / chunk_time)), desc="Recording"):
            chunk = stream.read(int(self.rate * chunk_time))
            self.buffer.append(chunk)

        if finish_record_event:
            finish_record_event.set()  # 向外部传达录音已结束的信息
        stream.stop_stream()
        stream.close()

        return self.buffer

    def record_auto(self, interrupt_event=None, finish_record_event=None, silence_threshold=200, max_silence_second=1):
        r"""根据外部环境的声音强度，智能录制有声音的片段。
        连续收到start_thr个有声块(chunk)就认为有人声，开始正式录音（这start_thr个块的数据也会保留）；
        开始录音后连续收到stop_thr个无声块就认为录音结束，退出录音；
        在一次录音过程中累计收到超过exit_thr个无声块时认为本次录音全程静音，退出录音。
        Args:
            interrupt_event (threading.Event): 打断事件，用于支持外部中断录音过程
            finish_record_event (threading.Event): 结束录制事件，用来向外部传达信息
            silence_threshold (int): deprecated
            max_silence_second (int): deprecated
        Return:
            buffer (list): 录制的音频，每个元素是一个块chunk，buffer[-1]是最新录制的数据。
                整段录制时才会用到，流式录制不需要访问这个返回值
            no_sound (bool): True表示本次录制没有录到声音，False表示录到了声音
        """
        assert self.channels == 1, "VAD only support mono channel!"
        if finish_record_event:
            finish_record_event.clear()  # 先复位事件
        t1 = time.time()
        
        # width = pyaudio.get_sample_size(self.pformat)
        # buffer_window_len = int(self.rate / self.chunk_length * max_silence_second)
        self.buffer = []
        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=self.chunk_length)

        vad = webrtcvad.Vad(mode=3)  # [0, 3]，0最容易给出是语音的判断，3最严格
        start_count = 0
        stop_count = 0
        exit_count = 0
        start_thr = 10  # 300ms
        stop_thr = 50  # 1.5s
        exit_thr = 333  # 10s
        while True:
            if interrupt_event and interrupt_event.is_set():
                # 如果收到外部的打断信号，则停止录音，将当前录下的部分输出
                break

            # stream.read()的参数是n_frames，不是n_bytes
            chunk = stream.read(self.chunk_length)

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
                    # NOTE: 这里不能写self.buffer = []，因为这个buffer给了ASR模块当浅拷贝，赋空值就是新的存储空间了
                    self.buffer.clear()
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
        self.no_sound = exit_count >= exit_thr

        if finish_record_event:
            finish_record_event.set()  # 向外部传达录音已结束的信息
        stream.stop_stream()
        stream.close()

        self.cost_time = time.time() - t1  # 流式录制翻译没办法直接获取录音模块花费的时间，只好出此下策

        return self.buffer, self.no_sound


class Player:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=8000, chunk_length=1024):
        # voiceai的TTS模块目前只支持8000采样率，在播放音频时需要注意保持采样率一致
        self.pformat = pformat
        self.channels = channels
        self.rate = rate
        self.chunk_length = chunk_length
        self.chunk_bytes = chunk_length * channels * pyaudio.get_sample_size(self.pformat) # 一个块CHUNK里包含的字节数

        self.audio = pyaudio.PyAudio()

        self.wav_data = None
        self.seek = 0

        self.play_frames = 0  # 仅用于配合tqdm_iterator使用，表示stream的while播放循环中已经播放了几帧数据
        self.t1 = -1
        self.start_time = -1

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

    def _callback(self, in_data, frame_count, time_info, status):
        start = self.seek
        self.play_frames += 1
        self.seek += int(frame_count * self.channels * pyaudio.get_sample_size(self.pformat))
        if self.seek < len(self.wav_data):
            # paContinue表示后面还有数据
            return self.wav_data[start:self.seek], pyaudio.paContinue
        elif start < len(self.wav_data):  #  此块不足一整块，是最后一个块
            ret = self.wav_data[start:]
            # pad the last frame with zero to chunk_length and put signal pacontinue,
            # or the pyaudio would not play it.
            ret += b"\x00" * (self.chunk_bytes - len(ret))
            # 完整且状态为paContinue的块才会被播放，这里文档是说最后一个块用paComplete，但是那样这个块播放不了，
            # 我们还是要用paContinue并把这个块补全到完整长度
            return ret, pyaudio.paContinue
        else:
            # paComplete表示这是音频数据的最后一个块block
            return b"", pyaudio.paComplete

    def _callback_ws(self, in_data, frame_count, time_info, status):
        # TODO: 看看有没有办法让TTS模块一次收到刚好一个CHUNK的数据，这样就不用做bytes的相加操作，直接在list里取，
        # 这样才快而且省内存
        # 因为网络原因导致现在没有数据可播（实践中出现且仅出现在最开头），但还有一些数据在路上
        if self.seek >= len(self.ws.wav_data) and not self.ws.finish_tts:
            return b"\x00" * self.chunk_bytes, pyaudio.paContinue
        # 流式TTS播音没办法直接获取从准备播音到开始播音（也就是收到第一帧数）据花费的时间，只好出此下策
        if self.start_time == -1:
            self.start_time = time.time() - self.t1  # t1是在外部启动TTS的时候赋值的，没在类内赋值

        start = self.seek
        self.play_frames += 1
        self.seek += int(frame_count * self.channels * pyaudio.get_sample_size(self.pformat))

        if self.seek < len(self.ws.wav_data):
            # paContinue表示后面还有数据
            return self.ws.wav_data[start:self.seek], pyaudio.paContinue
        elif start < len(self.ws.wav_data):  #  此块不足一整块，是最后一个块
            ret = self.ws.wav_data[start:]
            # pad the last frame with zero to chunk_length and put signal pacontinue,
            # or the pyaudio would not play it.
            ret += b"\x00" * (self.chunk_bytes - len(ret))
            # 完整且状态为paContinue的块才会被播放，这里文档是说最后一个块用paComplete，但是那样这个块播放不了，
            # 我们还是要用paContinue并把这个块补全到完整长度
            return ret, pyaudio.paContinue
        else:
            # paComplete表示这是音频数据的最后一个块block
            return b"", pyaudio.paComplete

    def play_unblock(self, wav_data, wakeup_event=None):
        r"""非阻塞地播放一个音频流，期间允许被打断
        Args:
            wav_data (bytes): 音频流二进制数据
            wakeup_event (threading.Event()): 唤醒事件，用于打断音频播放
        """
        self.wav_data = wav_data
        self.seek = 0  # 文件指针

        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True,
                                 frames_per_buffer=self.chunk_length,
                                 stream_callback=self._callback)
        stream.start_stream()

        # tqdm_iterator仅用于展示播音过程，不想看可以去掉
        # NOTE: 目前tqdm_iterator还有未知问题，有时候正常有时候会在next(tqdm_iterator)处抛出StopIteration异常，只好先不打开
        # n_chunks = round(len(self.wav_data) / self.chunk_bytes + 0.5)  # 等同于math.ceil，不想为了这个多import一个math
        # tqdm_iterator = iter(tqdm(range(n_chunks), "播音中"))

        # self.play_frames, prev = 0, 0
        while stream.is_active():
            # 在播放音频的时候，每0.1s检测一次是否被唤醒，如果被唤醒则停止播音
            if wakeup_event and wakeup_event.is_set():
                break
            time.sleep(0.1)
            # for _ in (range(self.play_frames - prev)):
            #     next(tqdm_iterator)
            # prev = self.play_frames
        # 把最后剩下的一点点走完，这里因为是自己iter(tqdm())可能没用对，由于某种原因会剩下一点点
        # for _ in tqdm_iterator:
        #     pass

        # tx2开发板get_output_latency是0.256s
        time.sleep(stream.get_output_latency())
        stream.stop_stream()
        stream.close()

    def play_unblock_ws(self, ws, interrupt_event=None):
        r"""非阻塞地播放一个音频流，期间允许被打断。这里传进来的不是音频数据，而是一个WebSocket对象，它
        在一边加载数据的同时，这边也在持续播放音频
        Args:
            ws (websocket.WebSocketApp): 用来获取音频数据流的对象
            interrupt_event (threading.Event()): 中断事件，用于打断音频播放
        """
        self.ws = ws
        self.seek = 0  # 文件指针

        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True,
                                 frames_per_buffer=self.chunk_length,
                                 stream_callback=self._callback_ws)
        self.start_time = -1
        stream.start_stream()

        while len(self.ws.wav_data) <= 0:  # 等待TTS服务器给到第一个数据包
            time.sleep(0.1)

        while stream.is_active():
            if interrupt_event and interrupt_event.is_set():
                break
            time.sleep(0.1)

        # tx2开发板get_output_latency是0.256s
        # 在目前版本中我们把最后一个块也给了paContinue，它一定能播放完，不需要这个sleep也能正常工作
        # time.sleep(stream.get_output_latency())
        stream.stop_stream()
        stream.close()


def test_player():
    player = Player(rate=INPUT_RATE)
    with wave.open("./tmp2.wav", "rb") as wf:
        wav_data = wf.readframes(wf.getnframes())
    event = threading.Event()
    player.play_unblock(wav_data, event)


def test_record():
    r"""借助Player进行测试"""
    recorder = Recorder(rate=INPUT_RATE)
    player = Player(rate=INPUT_RATE)

    # 测试阻塞式录制
    print("Blocking recording:")
    buffer = recorder.record(5)

    print("Finish recording, now playing...")
    player.play_unblock(b"".join(buffer))

    # 测试非阻塞式录制
    print("Unlocking recording:")
    finish_record_event = threading.Event()
    s = threading.Thread(target=recorder.record,
                         kwargs=({
                             "record_seconds": 5,  # 录制5s
                             "finish_record_event": finish_record_event
                         }))
    s.setDaemon(True)
    s.start()
    while not finish_record_event.is_set():
        time.sleep(0.2)

    print("Finish recording, now playing...")
    player.play_unblock(b"".join(recorder.buffer))


def test_record_auto():
    r"""借助Player进行测试"""
    # record_auto only support mono channel
    recorder = Recorder(rate=INPUT_RATE, channels=1)
    player = Player(rate=INPUT_RATE, channels=1)

    # 测试阻塞式录制
    print("Blocking recording:")
    buffer, _ = recorder.record_auto()

    print("Finish recording, now playing...")
    player.play_unblock(b"".join(buffer))

    # 测试非阻塞式录制
    print("Unblocking recording:")
    finish_record_event = threading.Event()
    s = threading.Thread(target=recorder.record_auto,
                         kwargs=({
                             "finish_record_event": finish_record_event
                         }))
    s.setDaemon(True)
    s.start()
    while not finish_record_event.is_set():
        time.sleep(0.2)

    print("Finish recording, now playing...")
    player.play_unblock(b"".join(recorder.buffer))


def test_play_unblock_ws():
    text = "妈妈给六岁的儿子出算术题做。“你一共有六个苹果，爸爸拿走两个，妈妈拿走四个，你还剩几个苹果？”儿子听后很激动说：“我真的是亲生的吗？”"
    tts = TTSBiaobei()
    player = Player(rate=OUTPUT_RATE)
    ws = tts.start_tts(text)
    player.play_unblock_ws(ws)


def test_listener():
    listener = Listener(FORMAT, CHANNELS, INPUT_RATE, LISTENER_CHUNK_LENGTH, LISTEN_SECONDS)
    listener.listen()
    time.sleep(20)
    # frames = listener.buffer[-int(INPUT_RATE / LISTENER_CHUNK_LENGTH * LISTEN_SECONDS):]
    listener.stop()


if __name__ == "__main__":
    # test_record()
    test_record_auto()
    # test_player()
    # test_play_unblock_ws()
    # test_listener()
