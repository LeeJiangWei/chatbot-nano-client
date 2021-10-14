import wave
import time
import multiprocessing
from utils.transcribe_streaming import read_file_in_chunks, streaming_request_iterable, str2bool
import grpc
from voiceai.cloud.speech.v1 import cloud_speech_pb2_grpc
from voiceai.cloud.speech.v1 import cloud_speech_pb2

import audioop
import pyaudio
from tqdm import tqdm
import webrtcvad
import argparse
import sys
from math import ceil

CHUNK_LENGTH = 1024
RECORDER_CHUNK_LENGTH = 30  # 一个块=30ms的语音
LISTENER_CHUNK_LENGTH = 1000  # 一个块=1s的语音


class Listener:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=16000, chunk_length=LISTENER_CHUNK_LENGTH,
                 listen_seconds=1):
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
        stop_thr = 66  # 2s
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


class Recognizer:
    def __init__(self, args, pformat=pyaudio.paInt16, channels=1, rate=8000, chunk_length=RECORDER_CHUNK_LENGTH):
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
        self.args = args

    def __del__(self):
        self.audio.terminate()

    @staticmethod
    def read_chunks(stream: pyaudio.Stream, chunk_size):
        """Lazy function (generator) to read a file piece by piece.
        Default chunk size: 1k."""
        while True:
            data = stream.read(1)
            if not data or stream.is_stopped():
                break
            yield data

    def _callback(self, in_data, frame_count, time_info, status):
            pass

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
        start = time.time()
        chunk_size = self.rate

        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=chunk_size)
        stream_generator = self.read_chunks(stream, chunk_size)

        options = [("grpc.keepalive_permit_without_calls", 1),
                   ("grpc.http2.max_pings_without_data", 0)]

        with grpc.insecure_channel(self.args.serverAddress, options) as channel:
            stub = cloud_speech_pb2_grpc.SpeechStub(channel)

            requests = (cloud_speech_pb2.StreamingRecognizeRequest(audio_content=chunk)
                        for chunk in stream_generator)
            model_selction = cloud_speech_pb2.ModelSelection(
                model_name=self.args.modelName,
                version_label=self.args.versionLabel
            )
            config = cloud_speech_pb2.RecognitionConfig(
                model_selection=model_selction,
                language_code='zh-CN',
                encoding=cloud_speech_pb2.RecognitionConfig.AudioEncoding.LINEAR16,
                enable_automatic_punctuation=self.args.addPunctuation,
                sample_rate_hertz=self.args.sampleRate,
                vendor_proprietary_config="--convert-numbers=" + str(
                    self.args.convertNumbers).lower() + " --oral2written=" + str(
                    self.args.oral2written).lower() + " --segmentation=" + str(
                    self.args.segmentation).lower() + " --need-timeinfo=" + str(
                    self.args.needTimeinfo).lower() + " --pause-time=" + str(
                    self.args.pauseTime) + " --acoustic-scale=" + str(
                    self.args.acousticScale))
            streaming_config = cloud_speech_pb2.StreamingRecognitionConfig(config=config)
            responses = stub.StreamingRecognize(streaming_request_iterable(streaming_config, requests))

            recog_time = time.time()
            print("start")
            for response in responses:
                print(response)
                if response.speech_event_type == 1:
                    print("terminate")
                    channel.close()
                    break
                # Once the transcription has settled, the first result will contain the
                # `is_final` result. The other results will be for subsequent portions of
                # the audio.
                for result in response.results:
                    # 注：在实际应用项目中，App代码可以只处理`is_final`为true（表示这个已是稳定的
                    # 转写结果）的response，忽略掉`is_final`为false的response。
                    # 如果你想track转写的中间过程，可不用判断`is_final`，如此把转写中间过程的reponses
                    # 全部print出来作为debug用途。
                    if result.is_final:
                        # print('Is Final: {}'.format(result.is_final))
                        # print('Stability: {}'.format(result.stability))
                        alternatives = result.alternatives
                        # The alternatives are ordered from most likely to least.
                        for alternative in alternatives:
                            # `alternative.confidence`是这个转写结果的置信度, the confidence
                            # print('Confidence: {}'.format(alternative.confidence))
                            # `alternative.transcript`是转写结果, the asr result
                            # print(u'Transcript: {}'.format(alternative.transcript))
                            print(u'{}'.format(alternative.transcript))
                        stream.stop_stream()
                        stream.close()
                        end = time.time()
                        print(recog_time - start , end - recog_time)
                        return alternative.transcript




class Player:
    def __init__(self, pformat=pyaudio.paInt16, channels=1, rate=8000):
        # should match TTS module
        self.pformat = pformat
        self.channels = channels
        self.rate = rate

        self.audio = pyaudio.PyAudio()

        self.wav_data = None
        self.wakeup_event = None
        self.seek = 0

        self.play_frames = 0  # 仅用于配合tqdm_iterator使用，表示stream的while播放循环中两次time.sleep()间隔播放了几帧数据

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
        time.sleep(stream.get_output_latency() * 2)
        stream.stop_stream()
        stream.close()

    def _callback(self, in_data, frame_count, time_info, status):
        start = self.seek
        self.play_frames += 1
        # wave_file 的读取就是双通道的, 但通道音频仅仅只是将另一通道置零, 而且python的bytes是16位的
        #  frame_count * 2 channels *  sample_size (16bits) / 2 (16bit)
        self.seek += int(frame_count * 2 * pyaudio.get_sample_size(self.pformat) / 2)
        print(start, self.seek)
        if self.seek < len(self.wav_data):
            # print("call")

            print(len(self.wav_data[start:self.seek]))
            return self.wav_data[start:self.seek], pyaudio.paContinue
        elif start < len(self.wav_data):
            ret = self.wav_data[start:]
            # pad the last frame with zero to chunk_length and put singal pacontinue,
            # or the pyaudio would not play it. f**k the pyaudio
            ret = ret + b"\x00" * (frame_count * 2 - len(ret))
            print(len(ret))
            return ret, pyaudio.paContinue
        else:
            return b"", pyaudio.paComplete

    def play_unblock(self, wav_data, wakeup_event):
        r"""非阻塞地播放一个音频流，期间允许被打断
        Args:
            wav_data (bytes): 音频流二进制数据
            wakeup_event (multiprocessing.Event()): 唤醒事件，用于打断音频播放
        """
        self.wav_data = wav_data
        self.wakeup_event = wakeup_event
        self.seek = 0  # 文件指针

        print(len(self.wav_data))
        stream = self.audio.open(format=self.pformat,
                                 channels=self.channels,
                                 rate=self.rate,
                                 output=True,
                                 frames_per_buffer=CHUNK_LENGTH,
                                 stream_callback=self._callback)
        stream.start_stream()

        # tqdm_iterator仅用于展示播音过程，不想看可以去掉
        chunk_bytes = CHUNK_LENGTH * self.channels * pyaudio.get_sample_size(self.pformat)  # 一个CHUNK所包含的字节数
        # n_chunks = round(len(self.wav_data) / chunk_bytes + 0.5)  # 等同于math.ceil，不想为了这个多import一个math
        n_chunks = ceil(len(self.wav_data) / chunk_bytes)
        # tqdm_iterator = iter(tqdm(range(n_chunks), "播音中"))

        while stream.is_active():
            self.play_frames = 0
            # 在播放音频的时候，每0.1s检测一次是否被唤醒，如果被唤醒则停止播音
            if wakeup_event.is_set():
                break
            time.sleep(0.1)
            # for _ in (range(self.play_frames)):
                # next(tqdm_iterator)
        # 把最后剩下的一点点走完
        # for _ in tqdm_iterator:
        #     pass

        time.sleep(stream.get_output_latency() * 2)
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

#
# if __name__ == "__main__":
#     print("Locale/Code Page of your OS environment is {}".format(sys.stdout.encoding))
#
#     parser = argparse.ArgumentParser(
#         description=__doc__,
#         formatter_class=argparse.RawDescriptionHelpFormatter)
#
#     # ASR服务器地址和端口 ASR Server address and port
#     parser.add_argument('--server-address', dest='serverAddress', metavar='host:port', action='store',
#                         default='voiceai.sution.top:50051', required=False,
#                         help='The server address in the format of host:port')
#     # wav语音文件的采样率 the sample rate of the speech
#     parser.add_argument('--sample-rate', dest='sampleRate', action='store', default=16000, type=int,
#                         choices=[16000, 8000], required=False, help='Sample rate of the audio data.')
#     # 是否加标点符号  do you need punctuation or not
#     parser.add_argument('--add-punctuation', dest='addPunctuation', type=str2bool, nargs='?',
#                         default=False, metavar='true, false', action='store',
#                         help='Add punctuation to the results or not.')
#     # 是否将数字转换为阿拉伯数字  convert numbers to arabic numbers
#     parser.add_argument('--convert-numbers', dest='convertNumbers', type=str2bool, nargs='?',
#                         default=True, metavar='true, false', action='store', help='Convert numbers to arabic format.')
#     # 是否将口语转换为书面语 convert oral text to written text
#     parser.add_argument('--oral2written', dest='oral2written', type=str2bool, nargs='?',
#                         default=False, metavar='true, false', action='store',
#                         help='Cconvert oral text to written text format.')
#     # 是否分词  segmentation result or not
#     parser.add_argument('--segmentation', dest='segmentation', type=str2bool, nargs='?',
#                         default=False, metavar='true, false', action='store', help='Segment the results or not.')
#     # 是否需要每个paragraph的时间信息  return the starting and ending time of every paragraph
#     parser.add_argument('--need-timeinfo', dest='needTimeinfo', type=str2bool, nargs='?',
#                         default=True, metavar='true, false', action='store', help='Need time information or not.')
#     # 静音检测时长  pause time of every paragraph
#     parser.add_argument('--pause-time', dest='pauseTime', type=int, default=250,
#                         help='The pause time that will force the decoder to reset and return end_of_paragraph, the unit is 0.01 second (e.g pause-time=250 means 2.5 seconds) and valid range is [50, 500].')
#     # --acoustic-scale
#     parser.add_argument('--acoustic-scale', dest='acousticScale', type=float, default=1.0, help='The acoustic scale.')
#     # 选择使用的模型名 the model name used for transcribing
#     parser.add_argument('--model-name', dest='modelName', metavar='susie-asr-8k', action='store', default='',
#                         required=False, help='The model name will be used for the transcribing')
#     # 选择使用的模型版本标签 the version label of the model name used for transcribing
#     parser.add_argument('--version-label', dest='versionLabel', metavar='stable', action='store', default='',
#                         required=False, help='The version label of the model name will be used for the transcribing.')
#
#     args = parser.parse_args(["--server-address=voiceai.sution.top:50051",
#                               "--sample-rate=8000",
#                               "--add-punctuation=true",
#                               "--convert-numbers=true",
#                               "--oral2written=true",
#                               "--segmentation=false",
#                               "--need-timeinfo=true",
#                               "--pause-time=250",
#                               "--model-name=susie-asr-8k",
#                               "--version-label=stable"])
#
#     recg = Recognizer(args)
#     print(recg.record_auto())


if __name__ =="__main__":
    player = Player(rate=16000)
    with wave.open("./juice3.wav") as wf:
        print(wf.getnframes())
        wave_data = wf.readframes(wf.getnframes())
        player.play_unblock(wave_data, multiprocessing.Event())
