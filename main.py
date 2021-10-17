# -*- coding: utf-8 -*-
import os
import sys
import logging
import socket
import threading
import time
import random
from PIL import Image, ImageDraw, ImageFont

import pyaudio
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 把tensorflow的日志等级降低，不然输出一堆乱七八糟的东西
import tensorflow as tf

from waker import Waker
from audiohandler import Listener, Recorder, Player
from utils.utils import bytes_to_wav_data, save_wav, get_answer, synonym_substitution
from utils.asr_utils import ASRVoiceAI
from utils.vision_utils import get_color_dict
from utils.package_utils import (Package, groupSendPackage, transform_for_send, client_service)
from api import (str_to_wav_bin_unblock, str_to_wav_bin,
                 VoicePrint)
from vision_perception import K4aCamera, NormalCamera
from vision_perception.client_for_voice import InfoObtainer

######################################################################################################
# 1. socket (127.0.0.1, 5588)                                                                        #
# 2. client call function:                                                                           #
#       1) start speech                                                                              #
#       2) close speech                                                                              #
#       3) exit reading                                                                              #
#       4) heart-beat(per 10s)                                                                       #
# 3. server return data:                                                                             #
#       1) real time detection result                                                                #
#       2) real time state (start speaking, exit speaking, start reading, exit reading, computing)   #
#       3) voice to word result (including successful result and other unexcepted error)             #
#       4) response word result                                                                      #
######################################################################################################

# real time state:
#   INITIALIZATION\EXIT-SPEAKING\EXIT-READING: 0
#   START-SPEAKING: 1
#   START-READING:  2
#   COMPUTING:      3

SOCKET = ("0.0.0.0", 5588)  # 后端监听的端口

HOST = "222.201.134.203"
PORT = 17000
PORT_INFO = 17001
scene_cam = K4aCamera(HOST, PORT)
emotion_cam = NormalCamera(0, "emotion", HOST, PORT)
I = InfoObtainer(HOST, PORT_INFO)

RECORDER_CHUNK_LENGTH = 30  # 一个块=30ms的语音
LISTENER_CHUNK_LENGTH = 1000  # 一个块=1s的语音
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 8000
LISTEN_SECONDS = 1
EXPECTED_WORD = "miya"

PLOT = False

BBOX_COLOR_DICT = get_color_dict()

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)


class MainProcess(object):
    def __init__(self):
        self.clients = {}
        self.state = 0
        self.detection_result = ''
        return

    def __setattr__(self, key, value):
        # 巧妙的设计，在服务端对self.state和self.detection_result进行赋值操作时，把新值广播给所有的客户端
        if key in self.__dict__ and value == self.__dict__[key]:
            return
        self.__dict__[key] = value
        if key == 'state':
            # group send real time state
            groupSendPackage(Package.real_time_state(value), self.clients)
        elif key == 'detection_result':
            groupSendPackage(Package.real_time_detection_result(value), self.clients)
        return

    def start_speech(self):
        if self.state == 0:
            self.state = 1
        return

    def close_speech(self):
        if self.state == 1:
            self.state = 0
        return

    def close_reading(self):
        if self.state == 2:
            self.state = 0
        return

    def main(self):
        r"""
        时序图：─│┌┐└┘├┴┬┤┼╵
                                               ┌─没检测到唤醒词──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        初始阶段：listener.listen()录制用户语音 ─┴─检测到唤醒词── 发送"你好米娅"文本到前端 ─── 进行声纹识别 ─── 发送欢迎文本到前端 ─── 播放欢迎语 ─── wakeup_event.set() ─── 继续listener.listen()
                                                                                    │ 主进程可以在record_auto()结束时、播放语音前或者播放语音时通过说唤醒词退出互动阶段
        互动阶段：wakeup_event.set() ─── 进入interact_process循环 ─── recorder.record_auto()录制用户语音 ─── I.obtain()向视觉模块获取信息 ─── wav_bin_to_str()语音转文字 ─── get_answer()获取响应文本 ─── 边转音频边播放 ─── wakeup_event.clear()
        """
        self.wakeup_event = threading.Event()
        self.is_playing = False
        self.player_exit_event = threading.Event()
        self.all_exit_event = threading.Event()
        self.haddata = False  # 用于InfoObtainer的共享变量
        self.finish_stt_event = threading.Event()  # 用于流式STT中判断STT是否已经结束

        inter_proc = threading.Thread(target=self.interact, args=())
        inter_proc.setDaemon(True)  # 设置成守护进程，不然ctrl+C退出main()子进程还在，程序依然卡死
        inter_proc.start()

        listener = Listener(FORMAT, CHANNELS, INPUT_RATE, LISTENER_CHUNK_LENGTH, LISTEN_SECONDS)
        vpr = VoicePrint()
        self.player = Player(rate=OUTPUT_RATE)
        waker = Waker(EXPECTED_WORD)
        # recorder跟asr用于interact()
        self.recorder = Recorder(FORMAT, CHANNELS, INPUT_RATE, RECORDER_CHUNK_LENGTH)
        self.asr = ASRVoiceAI(INPUT_RATE)

        with tf.compat.v1.Session() as sess:
            # main loop
            while True:
                listener.listen()
                # keyword spotting loop
                print("Listening...")
                while not waker.waked_up():
                    # frames包含了最近LISTEN_SECONDS内的音频数据
                    frames = listener.buffer[-int(INPUT_RATE / LISTENER_CHUNK_LENGTH * LISTEN_SECONDS):]
                    wav_data = bytes_to_wav_data(b"".join(frames), FORMAT, CHANNELS, INPUT_RATE)
                    waker.update(wav_data, sess, PLOT)

                waker.reset()  # 重置waker的置信度等参数，使其下轮循环能重新进入内层while循环，等待下一次唤醒
                scene_cam.send_single_image()
                self.haddata = False  # False要求重新向视觉模块获取视觉信息
                listener.stop()
                print("WAKEUP!")

                recognized_str = "你好米娅"
                # 将用户输入的语音转换成文字的结果群发给每个前端
                groupSendPackage(Package.voice_to_word_result(recognized_str), self.clients)
                # 如果STT（语音转文字）异常，通过置success=False向前端传达出错了的信号
                # groupSendPackage(Package.voice_to_word_result("STT出错", success=False), self.clients)  # 具体错误可以具体写
                # self.state = 0

                # save_wav(b"".join(frames), "tmp.wav")  # debug临时语句，保存原本音频流以确保play之前的部分都正常运行

                # wakeup
                if not self.is_playing:
                    t1 = time.time()
                    spk_name = vpr.get_spk_name(wav_data)
                    print(f"声纹识别耗费时间为：{time.time() - t1:.2f}秒")
                    response_str = spk_name + "你好!"

                    # 针对用户情绪，在原有的欢迎语上添加对应的句子
                    emotion_cam.send_single_image()
                    data = {"require": "emotion"}
                    result = I.obtain(data, False)
                    print("情绪识别结果：", result["emotion"])
                    welcome_word_suffix = {  # 分开心、不开心、平静三类
                        "neutral": ["你别绷着个脸嘛，笑一笑十年少", "你冷峻的脸庞让我着迷", "你就是这间房最酷的仔", "高冷就是你的代名词 "],
                        "happy": ["你看起来心情不错", "你看起来很开心，我猜是捡到钱了", "你笑起来真好看", "你的笑容让我沉醉"],
                        "unhappy": ["你看起来似乎不太开心", "你的心情有些低落呢，给你讲个笑话好不好", "别生气啦，来杯咖啡压压惊", "压力大就要学会放松心情"]
                        }
                    if result["emotion"] in ["neutral"]:  # calm
                        response_str += random.choice(welcome_word_suffix["neutral"])
                    elif result["emotion"] in ["happiness"]:  # happy
                        response_str += random.choice(welcome_word_suffix["happy"])
                    elif result["emotion"] in ["surprise", "fear", "digust", "sadness", "anger"]:  # unhappy
                        response_str += random.choice(welcome_word_suffix["unhappy"])
                    elif result["emotion"] == "no_face":
                        pass
                    else:
                        raise ValueError(f"Unrecognized emotion: {result['emotion']}")

                    wav = str_to_wav_bin(response_str)
                # interrupt
                elif self.is_playing:
                    self.wakeup_event.set()
                    response_str = "我在。"
                    wav = str_to_wav_bin(response_str)
                    self.player_exit_event.wait()

                # 将智能系统回答的文本群发给每个前端
                groupSendPackage(Package.response_word_result(response_str), self.clients)

                self.state = 2  # speaking
                self.player_exit_event.clear()
                self.player.play_unblock(wav)
                self.wakeup_event.set()

                if self.all_exit_event.is_set():  # 正常退出，不使用ctrl+C退出，不然相机老是出幺蛾子
                    print("退出主程序。")
                    break

    def interact(self):
        while True:
            print("Wait to be wakeup...")
            self.wakeup_event.wait()
            self.wakeup_event.clear()
            while True:
                self.is_playing = True
                logger.info("Start recording...")
                self.state = 1  # recording
                wav_list, no_sound = self.recorder.record_auto()
                logger.info("Stop recording.")
                if no_sound:
                    logger.info("No sound detected, conversation canceled.")
                    break

                # 如果在录音的过程中收到唤醒词（比如唤醒之后说的还是唤醒词），录音结束后将来到这里，应当不对录音内容做互动处理，跳出互动阶段
                if self.wakeup_event.is_set():
                    self.wakeup_event.clear()
                    break

                logger.info("Waiting server...")
                self.state = 3  # computing

                # scene_cam.send_single_image()
                # self.haddata = False  # False要求重新向视觉模块获取视觉信息
                # time.sleep(0.1)  # 给视觉模块时间重置信息
                # TODO: I.obtain跟wav_bin_to_str是可以并行的，vodiceai转文字的时间比后面改成并行
                data = {"require": "attribute"}
                result = I.obtain(data, self.haddata)
                self.haddata = True  # 获得过一次视觉信息之后，在下次重新唤醒之前就不需要重复获取了
                
                with open("visual_info.txt", "w") as fp:
                    fp.write("时间戳:" + result["timestamp"] + "\n")
                    for item in result["attribute"]:
                        fp.write(str(item) + "\n")

                img = Image.open(scene_cam.savepath).convert("RGB")
                drawer = ImageDraw.ImageDraw(img)
                fontsize = 13
                font = ImageFont.truetype("Ubuntu-B.ttf", fontsize)
                for attr in result["attribute"]:
                    # text参数: 锚点xy，文本，颜色，字体，锚点类型(默认xy是左上角)，对齐方式
                    # anchor含义详见https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html
                    drawer.text(attr["bbox"][:2], attr["category"], fill=BBOX_COLOR_DICT[attr["category"]], font=font, anchor="lb", align="left")
                    # rectangle参数: 左上xy右下xy，边框颜色，边框厚度
                    drawer.rectangle(attr["bbox"][:2] + attr["bbox"][2:], fill=None, outline=BBOX_COLOR_DICT[attr["category"]], width=2)
                img.save("tmp2.jpg")
                with open("tmp2.jpg", "rb") as fp:
                    # 赋值后就会自动把图像发给前端
                    self.detection_result = transform_for_send(fp.read())

                t0 = time.time()
                wav_data = bytes_to_wav_data(b"".join(wav_list))
                recognized_str = self.asr.trans(wav_data)
                print("近义词替换前：", recognized_str)
                recognized_str = synonym_substitution(recognized_str)
                print("近义词替换后：", recognized_str)

                if len(recognized_str) == 0:
                    recognized_str = "(无人声)"
                t1 = time.time()
                print("recognition:", t1 - t0)
                if recognized_str == "(无人声)" or "没事" in recognized_str:
                    break
                recognized_str = recognized_str.replace("~", "")
                groupSendPackage(Package.voice_to_word_result(recognized_str), self.clients)

                start = time.time()
                response_word, self.sentences = get_answer(recognized_str, result["attribute"])
                print("rasa:", time.time() - start)

                # TODO: 把退出主程序的功能做好，现在没有实现预期功能
                if recognized_str in ["退出。", "关机。"]:
                    # out-of-date
                    print("退出互动环节...")
                    self.all_exit_event.set()
                    time.sleep(10)  # 坐等主进程退出
                    break

                # 如果在准备回复内容的过程中收到唤醒词，回复内容准备好之后将来到这里，应当不继续播音，跳出互动阶段
                if self.wakeup_event.is_set():
                    self.wakeup_event.clear()
                    break

                self.state = 2  # speaking
                self.finish_stt_event.clear()
                self.wav_data_queue = []
                thread_stt = threading.Thread(target=str_to_wav_bin_unblock, args=(self.sentences, self.wav_data_queue, self.finish_stt_event))
                thread_stt.setDaemon(True)
                thread_stt.start()

                start = time.time()
                sent_response = False  # 是否已经发送过智能系统的回答文本给到前端
                # 当TTS尚未结束时，检查STT线程有没有生产wav data过来，有的话就拿去播放
                # TODO:加入唤醒打断
                while not self.finish_stt_event.is_set():
                    while self.wav_data_queue:
                        if not sent_response:
                            # 第一个句子翻译完开始播音时就可以把所有句子都打印在前端，因为把一句话说完需要时间，在用户看来
                            # 并不会觉得声音还没发出来字已经有了是违和的
                            groupSendPackage(Package.response_word_result(response_word), self.clients)
                            sent_response = True
                        wav_data = self.wav_data_queue.pop(0)
                        # self.player.play_unblock(w, self.wakeup_event)
                        self.player.play_unblock(wav_data)
                print("tts & play:", time.time() - start)

                # 如果在播音时收到唤醒词，应当先从play的播放循环中跳出，然后来到这里，跳出互动阶段
                if self.wakeup_event.is_set():
                    self.wakeup_event.clear()
                    break

            self.state = 0  # waiting
            self.player_exit_event.set()
            self.is_playing = False


def main():
    # 原先的main变成如今的main_process.main
    main_process = MainProcess()
    t = threading.Thread(target=main_process.main, args=())
    t.setDaemon(True)
    t.start()

    sock = socket.socket()
    # 该配置允许后端在断开连接后端口立即可用，也即没有TIME_WAIT阶段，调试必备
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(SOCKET)
    sock.listen(5)
    while True:
        recv_sock, recv_addr = sock.accept()
        t = threading.Thread(target=client_service, args=(recv_sock, recv_addr, main_process, ))
        t.setDaemon(True)
        t.start()


if __name__ == "__main__":
    main()
