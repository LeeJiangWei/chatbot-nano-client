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
from utils.utils import (bytes_to_wav_data, save_wav,
    get_answer, synonym_substitution, remove_punctuation)
from utils.asr_utils import ASRVoiceAI
from utils.tts_utils import TTSBiaobei
from utils.vision_utils import get_color_dict
from utils.package_utils import (Package, groupSendPackage, transform_for_send,
    client_service)
from api import VoicePrint, BaiduDialogue
from vision_perception import K4aCamera, NormalCamera
from vision_perception.client_for_voice import InfoObtainer
from vision_perception.client import get_visual_info

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
#   START-SPEAKING: 1  用户开始说话
#   START-READING:  2  机器开始播音
#   COMPUTING:      3  后端正在推理

SOCKET = ("0.0.0.0", 5588)  # 后端监听的端口

HOST = "222.201.134.203"
PORT = 17000
PORT_INFO = 17001

RECORDER_CHUNK_LENGTH = 480  # 一个块=30ms的语音
LISTENER_CHUNK_LENGTH = 1000
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 8000
LISTEN_SECONDS = 1  # 跟Waker模型相关，它只支持16000采样率，片段长1s，改了要重新训练模型
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
        # 前端发送"开始说话"指令时，后端会执行该函数
        # 暂时没有用到，不管
        if self.state == 0:
            self.state = 1
        return

    def close_speech(self):
        # 前端发送"停止说话"指令时，后端会执行该函数
        self.interrupt_event.set()
        if self.state == 1:
            self.state = 0
        return

    def close_reading(self):
        # 前端发送"停止录音"指令时，后端会执行该函数
        self.interrupt_event.set()
        if self.state == 2:
            self.state = 0
        return

    def init_backend(self):
        # 定义用于后端自己的变量，由于当前版本是在外头的main函数里面调用了MainProcess.main()，即使用一个实例也没用
        self.finish_stt_event = threading.Event()  # 用于流式STT中判断STT是否已经结束
        self.finish_record_event = threading.Event()  # 用于流式录音+流式ASR中判断是否已经结束录音
        self.interrupt_event = threading.Event()  # 用于判断是否收到键盘的打断信号

        self.visual_info = {}  # 用于后续接收语音信息，若不初始化会在get_visual_info时报错

        self.listener = Listener(FORMAT, CHANNELS, INPUT_RATE, LISTENER_CHUNK_LENGTH, LISTEN_SECONDS)
        self.vpr = VoicePrint()
        self.chatter = BaiduDialogue()
        self.player = Player(rate=OUTPUT_RATE)
        self.waker = Waker(EXPECTED_WORD)

        # 下面的对象都是用于interact()的
        self.scene_cam = K4aCamera(HOST, PORT)
        self.emotion_cam = NormalCamera(0, "emotion", HOST, PORT)
        self.I = InfoObtainer(HOST, PORT_INFO)
        self.recorder = Recorder(FORMAT, CHANNELS, INPUT_RATE, RECORDER_CHUNK_LENGTH)
        self.asr = ASRVoiceAI(INPUT_RATE)
        self.tts = TTSBiaobei()

    def main(self):
        r"""
        时序图：─│┌┐└┘├┴┬┤┼╵
                                               ┌─没检测到唤醒词──────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
        初始阶段：listener.listen()录制用户语音 ─┴─检测到唤醒词── 发送"你好米娅"文本到前端 ─── 进行声纹识别 ─── 发送欢迎文本到前端 ─── 播放欢迎语 ─── 互动阶段 ───────────── 继续listener.listen()
                                                                                                                                              └─主进程可以在录制用户语音时、播放回答─┘
                                                                                                                                                的音频时通过按下Esc键退出互动阶段
        互动阶段：recorder.record_auto()录制用户语音 ───┬───┬ 检测到有人说话 ─── asr.trans()流式语音转文字 ─── 发送识别出的文本到前端 ─── get_answer()获取响应文本 ─── tts.start_tts()文字转语音, 流式加载音频  ────┬─继续recorder.record_auto()录制用户语音
                  └─同时get_visual_info()获取视觉信息 ─┘   └ 没检测到有人说话 ─── 退出互动阶段                                                                       └─player.play_unblock_ws()非阻塞式播放音频─┘
        """
        # 由于未知原因，把init_backend放在__init__的位置时会导致tf报错，不知道多线程做了什么事情，或许是tf的东西不支持多线程？
        self.init_backend()

        with tf.compat.v1.Session() as sess:
            # main loop
            while True:
                self.listener.listen()
                # keyword spotting loop
                logger.info("Listening...")
                while not self.waker.waked_up():
                    # frames包含了最近LISTEN_SECONDS内的音频数据
                    frames = self.listener.buffer[-int(INPUT_RATE / LISTENER_CHUNK_LENGTH * LISTEN_SECONDS):]
                    wav_data = bytes_to_wav_data(b"".join(frames), FORMAT, CHANNELS, INPUT_RATE)
                    self.waker.update(wav_data, sess, PLOT)

                self.waker.reset()  # 重置waker的置信度等参数，使其下轮循环能重新进入内层while循环，等待下一次唤醒
                self.listener.stop()
                logger.info("WAKEUP!")

                # save_wav(b"".join(frames), "tmp.wav")  # debug临时语句，保存原本音频流以确保play之前的部分都正常运行

                # wakeup
                recognized_str = "你好米娅"
                # 将用户输入的语音转换成文字的结果群发给每个前端
                self.send_question_to_frontend(recognized_str)
                # 如果STT（语音转文字）异常，通过置success=False向前端传达出错了的信号
                # self.send_question_to_frontend("STT出错", success=False)  # 具体错误可以具体写
                # self.state = 0

                t1 = time.time()
                spk_name = self.vpr.get_spk_name(wav_data)
                logger.info(f"声纹识别耗费时间为：{time.time() - t1:.2f}秒")
                response_str = spk_name + "你好!"

                # 针对用户情绪，在原有的欢迎语上添加对应的句子
                self.emotion_cam.send_single_image()
                data = {"require": "emotion"}
                result = self.I.obtain(data, False)
                logger.info(f"情绪识别结果：{result['emotion']}")
                welcome_word_suffix = {  # 分开心、不开心、平静三类
                    "neutral": ["你别绷着个脸嘛，笑一笑十年少", "你冷峻的脸庞让我着迷", "你就是这间房最酷的仔", "高冷就是你的代名词", "嘤嘤嘤，不要这么冷漠嘛", "戳戳，笑一个"],
                    "happy": ["你看起来心情不错", "你看起来很开心，我猜是捡到钱了", "你笑起来真好看", "你的笑容让我沉醉"],
                    "unhappy": ["你看起来似乎不太开心", "你的心情有些低落呢，给你讲个笑话好不好", "别生气啦，来杯咖啡压压惊", "压力大就要学会放松心情", "怎么一副苦瓜脸，笑一个嘛", "有什么不开心的事情吗"]
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

                # 将智能系统回答的文本群发给每个前端
                groupSendPackage(Package.response_word_result(response_str), self.clients)

                self.state = 2  # robot speaking
                ws = self.tts.start_tts(response_str)
                self.interrupt_event.clear()
                self.player.play_unblock_ws(ws, self.interrupt_event)

                # 进入互动环节，这个版本互动跟等待唤醒是分开的，不会相互纠缠，也没有唤醒词打断功能，只支持按键打断
                self.interact()

    def interact(self, query_text=None):
        r"""
        Args:
            query_text (str): 仅用于调试，当其不为None时，不进行录音，直接以query_text作为询问句，提高调试速度
        """
        while True:
            # 在录音之前把图片发给视觉模块，录音结束差不多就能收到识别结果了（视觉模块处理时间2~3s）
            visual_proc = threading.Thread(target=get_visual_info, args=(self,))
            visual_proc.setDaemon(True)
            visual_proc.start()

            self.state = 1  # human speaking
            if query_text is None:
                logger.info("Start recording...")
                # 流式录制+流式翻译=流式录制翻译

                t1 = time.time()
                # 根据环境音量自动录制一段有声音的音频，支持外部打断
                self.interrupt_event.clear()  # 每次用一个事件之前都要先clear，防止是前面留下来的
                record_proc = threading.Thread(target=self.recorder.record_auto,
                    kwargs=({
                        "interrupt_event": self.interrupt_event,
                        "finish_record_event": self.finish_record_event
                    }))
                record_proc.setDaemon(True)
                record_proc.start()

                recognized_str = self.asr.trans_stream(self.recorder.buffer, self.finish_record_event)

                logger.info("Stop recording, waiting server...")
                self.state = 3  # computing

                if self.recorder.no_sound:
                    logger.info("No sound detected, conversation canceled.")
                    break
                
                logger.info(f"Recording: {self.recorder.cost_time:.2f}s")
                logger.info(f"ASR: {time.time() - t1 - self.recorder.cost_time:.2f}s")

                logger.info(f"原始文本：{recognized_str}")
                recognized_str = synonym_substitution(recognized_str)

                if len(remove_punctuation(recognized_str)) <= 1:  # 去掉一些偶然的噪声被识别为“嗯”等单字导致误判有人说话的情况
                    recognized_str = "(无人声)"
                if recognized_str == "(无人声)" or "没事" in recognized_str:
                    break
                recognized_str = recognized_str.replace("~", "")

            else:
                recognized_str = query_text

            self.send_question_to_frontend(recognized_str)
            recognized_str = remove_punctuation(recognized_str)
            logger.info(f"处理后文本：{recognized_str}")

            # 聊天api不需要依赖视觉信息，所以可以先开起来，再接着等视觉信息
            chat_proc = threading.Thread(target=self.chatter.run, args=(recognized_str,))
            chat_proc.setDaemon(True)
            chat_proc.start()

            t1 = time.time()
            visual_proc.join()  # 等待获取视觉信息的线程运行结束，也即收到视觉模块的回复
            logger.info(f"Get visual info: {time.time() - t1:.2f}s")
            # 获取回复文本跟画图相互独立，另外开一个线程
            draw_proc = threading.Thread(target=self.draw_visual_result, args=())
            draw_proc.setDaemon(True)
            draw_proc.start()

            start = time.time()
            response_word, self.sentences = get_answer(self.chatter, recognized_str, self.visual_info["attribute"])
            logger.info(f"Rasa: {time.time() - start:.2f}s")

            self.state = 2  # robot speaking
            self.send_response_to_frontend(response_word)
            if query_text:  # 如果是直接给出文本，得到一次答复之后就退出，因为重复问同一句话没有意义
                return response_word

            self.player.t1 = time.time()
            tts_proc = threading.Thread(target=self.tts.start_tts, args=(response_word,))
            tts_proc.setDaemon(True)
            tts_proc.start()
            self.interrupt_event.clear()
            self.player.play_unblock_ws(self.tts.ws, self.interrupt_event)
            logger.info(f"音频准备时间：{self.player.start_time:.2f}s")

        self.state = 0  # waiting

    def exit(self):
        # 当程序执行结束时，经常出现__del__没有被调用的情况，这个现象暂时无法找到原因，只能手动关闭k4a相机，避免这种情况发生
        # cv2相机不需要此操作；ctrl+C打断程序不需要此操作，不会卡住
        self.scene_cam.close()

    def draw_visual_result(self, imgpath=None):
        r"""本函数支持外部单独调用，传入非空的imgpath为图片路径"""
        with open("visual_info.txt", "w") as fp:
            fp.write("时间戳:" + self.visual_info["timestamp"] + "\n")
            for item in self.visual_info["attribute"]:
                fp.write(str(item) + "\n")

        if imgpath is None:
            imgpath = self.scene_cam.savepath
        img = Image.open(imgpath).convert("RGB")
        drawer = ImageDraw.ImageDraw(img)
        fontsize = 13
        font = ImageFont.truetype("Ubuntu-B.ttf", fontsize)
        for attr in self.visual_info["attribute"]:
            # text参数: 锚点xy，文本，颜色，字体，锚点类型(默认xy是左上角)，对齐方式
            # anchor含义详见https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html
            drawer.text(attr["bbox"][:2], f'{attr["category"]} {round(attr["confidence"] * 100 + 0.5)}', fill=BBOX_COLOR_DICT[attr["category"]], font=font, anchor="lb", align="left")
            # rectangle参数: 左上xy右下xy，边框颜色，边框厚度
            drawer.rectangle(attr["bbox"][:2] + attr["bbox"][2:], fill=None, outline=BBOX_COLOR_DICT[attr["category"]], width=2)
        img.save("tmp2.jpg")
        with open("tmp2.jpg", "rb") as fp:
            # 赋值后就会自动把图像发给前端
            self.detection_result = transform_for_send(fp.read())

    def send_question_to_frontend(self, text, success=True):
        r"""向前端群发用户提出的问题文本"""
        groupSendPackage(Package.voice_to_word_result(text, success=success), self.clients)

    def send_response_to_frontend(self, text, success=True):
        r"""向前端群发智能助理的回答文本"""
        groupSendPackage(Package.response_word_result(text, success=success), self.clients)


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
        # NOTE: 这里需要传main_process类，导致上面不能把main_process包进启动进程里面，而不包进去会导致如果在实例化
        # MainProcess的时候用了初始化了tf的计算图，在多线程的时候就会报错，所以二者无法兼得，后续看看有没有谁有什么
        # 好办法处理一下
        t = threading.Thread(target=client_service, args=(recv_sock, recv_addr, main_process, ))
        t.setDaemon(True)
        t.start()

    # 虽然现在这个程序是直接Ctrl+C断开的，如果后续能够运行到底，就必须执行exit()
    main_process.exit()


if __name__ == "__main__":
    main()
