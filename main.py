import os
import sys
import logging
import socket
import threading
import time
import json
from base64 import b64encode
import gzip
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import cv2  # NOTE: 由于未知原因，先import cv2再import pyaudio会导致相机启动时报k4aException，所以这两行有先后顺序要求
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 把tensorflow的日志等级降低，不然输出一堆乱七八糟的东西
import tensorflow as tf
from matplotlib import cm
import time

from waker import Waker
from audiohandler import Listener, Recorder, Player
from utils.utils import get_response, bytes_to_wav_data, save_wav
from utils.vision_utils import get_color_dict, transform_for_send
from api import VoicePrint, str_to_wav_bin
from vision_perception import VisionPerception
from vision_perception.client_for_voice import InfoObtainer
import multiprocessing as mp

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


SOCKET = ("0.0.0.0", 5588)
EXPIRE_TIME = 30

HOST = "222.201.134.203"
PORT = 17000
PORT_INFO = 17001
perception = VisionPerception(HOST, PORT)
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


global clients
clients = {}

images = [
    '../static/img_6113974560314588376',
    '../static/img_6519413793248989031'
]
#
# with open(images[0], 'rb') as f:
#     initial_image = b64encode(gzip.compress(f.read(), 6)).decode('utf-8')





class MainProcess(object):
    def __init__(self):
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
            groupSendPackage(Package.real_time_state(value))
        elif key == 'detection_result':
            groupSendPackage(Package.real_time_detection_result(value))
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
        互动阶段：wakeup_event.set() ─── 进入interact_process循环 ─── recorder.record_auto()录制用户语音 ─── 向视觉模块获取信息 ─── 结合视觉信息向语音模块获取回复的音频数据 ─── player.play_unblock()播放语音回复 ─── wakeup_event.clear()
        """
        self.wakeup_event = threading.Event()
        self.is_playing = False
        self.player_exit_event = threading.Event()
        self.all_exit_event = threading.Event()
        self.haddata = False  # 用于InfoObtainer的共享变量

        inter_proc = threading.Thread(target=self.interact, args=())
        inter_proc.setDaemon(True)  # 设置成守护进程，不然ctrl+C退出main()子进程还在，程序依然卡死
        inter_proc.start()

        listener = Listener(FORMAT, CHANNELS, INPUT_RATE, LISTENER_CHUNK_LENGTH, LISTEN_SECONDS)
        vpr = VoicePrint()
        self.player = Player(rate=OUTPUT_RATE)
        waker = Waker(EXPECTED_WORD)

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
                perception.send_single_image()
                self.haddata = False  # False要求重新向视觉模块获取视觉信息
                listener.stop()
                print("WAKEUP!")

                recognized_str = "你好米娅"
                # 将用户输入的语音转换成文字的结果群发给每个前端
                groupSendPackage(Package.voice_to_word_result(recognized_str))
                # 如果STT（语音转文字）异常，通过置success=False向前端传达出错了的信号
                # groupSendPackage(Package.voice_to_word_result("STT出错", success=False))  # 具体错误可以具体写
                # self.state = 0

                # save_wav(b"".join(frames), "tmp.wav")  # debug临时语句，保存原本音频流以确保play之前的部分都正常运行

                # wakeup
                if not self.is_playing:
                    t1 = time.time()
                    spk_name = vpr.get_spk_name(wav_data)
                    print(f"声纹识别耗费时间为：{time.time() - t1:.2f}秒")
                    response_str = spk_name + "你好!"
                    wav = str_to_wav_bin(response_str)
                # interrupt
                elif self.is_playing:
                    self.wakeup_event.set()
                    response_str = "我在。"
                    wav = str_to_wav_bin(response_str)
                    self.player_exit_event.wait()

                # 将智能系统回答的文本群发给每个前端
                groupSendPackage(Package.response_word_result(response_str))

                self.state = 2  # speaking
                self.player_exit_event.clear()
                self.player.play(wav)
                self.wakeup_event.set()

                if self.all_exit_event.is_set():  # 正常退出，不使用ctrl+C退出，不然相机老是出幺蛾子
                    print("退出主程序。")
                    break



        image_index = 0

        while True:
            if self.state == 1:  # 前端按下录音按钮，这边开始录音
                # 如果不主动退出，speech时间为10秒钟
                start = time.time()
                while self.state == 1:
                    if time.time() - start >= 10:
                        break
                    else:
                        time.sleep(1)
                    if 1 < time.time() - start < 2:
                        # 返回detection result
                        image_index += 1
                        image_index %= 2
                        with open(images[image_index], 'rb') as f:
                            self.detection_result = b64encode(gzip.compress(f.read(), 6)).decode('utf-8')
                #  说话结束，返回语音转文字结果
                random = (int(time.time() * 100) % 2 == 1)
                if random:
                    # group send voice_to_word error result
                    # 如果stt（语音转文字）异常，通过置success=False向前端传达出错了的信号
                    # groupSendPackage(Package.voice_to_word_result('异常错误', success=False))  # 具体错误可以具体写
                    # self.state = 0
                    continue
                else:
                    # 将用户输入的语音转换成文字的结果群发给每个前端
                    groupSendPackage(Package.voice_to_word_result(response_str))
                    pass
                # computing
                self.state = 3
                time.sleep(1)
                # start reading
                self.state = 2
                # group send response_word_result
                # 将智能系统回答的文本群发给每个前端
                groupSendPackage(Package.response_word_result(response_str))
                time.sleep(3)
                self.state = 0


    def interact(self):
        recorder = Recorder(FORMAT, CHANNELS, INPUT_RATE, RECORDER_CHUNK_LENGTH)
        while True:
            print("Wait to be wakeup...")
            self.wakeup_event.wait()
            self.wakeup_event.clear()
            while True:
                self.is_playing = True
                logger.info("Start recording...")
                self.state = 1  # recording
                wav_list, no_sound = recorder.record_auto()
                logger.info("Stop recording.")
                if no_sound:
                    logger.info("No sound detected, conversation canceled.")
                    break

                # 如果在录音的过程中收到唤醒词（比如唤醒之后说的还是唤醒词），录音结束后将来到这里，应当不对录音内容做互动处理，跳出互动阶段
                if self.wakeup_event.is_set():
                    self.wakeup_event.clear()
                    break

                wav_data = bytes_to_wav_data(b"".join(wav_list))
                logger.info("Waiting server...")
                self.state = 3  # computing

                # perception.send_single_image()
                # self.haddata = False  # False要求重新向视觉模块获取视觉信息
                # time.sleep(0.1)  # 给视觉模块时间重置信息
                data = {"require": "attribute"}
                result = I.obtain(data, self.haddata)
                self.haddata = True  # 获得过一次视觉信息之后，在下次重新唤醒之前就不需要重复获取了
                
                with open("visual_info.txt", "w") as fp:
                    fp.write("时间戳:" + result["timestamp"] + "\n")
                    for item in result["attribute"]:
                        fp.write(str(item) + "\n")

                img = Image.open(perception.savepath).convert("RGB")
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

                recognized_str, response_list, wav_list = get_response(wav_data, result["attribute"])
                logger.info("Recognize result: " + recognized_str)

                # haven't said anything but pass VAD.
                if len(recognized_str) == 0 or "没事" in recognized_str:
                    break
                groupSendPackage(Package.voice_to_word_result(recognized_str))

                # TODO: 把退出主程序的功能做好，现在没有实现预期功能
                if recognized_str in ["退出。", "关机。"]:
                    print("退出互动环节...")
                    self.all_exit_event.set()
                    time.sleep(10)  # 坐等主进程退出
                    break

                # 如果在准备回复内容的过程中收到唤醒词，回复内容准备好之后将来到这里，应当不继续播音，跳出互动阶段
                if self.wakeup_event.is_set():
                    self.wakeup_event.clear()
                    break

                self.state = 2  # speaking
                for r, w in zip(response_list, wav_list):
                    logger.info(r)
                    groupSendPackage(Package.response_word_result(r))
                    print("outer:", self.wakeup_event.is_set())
                    save_wav(w, "tmp.wav", rate=self.player.rate)  # debug临时语句，保存原本音频流以确保play之前的部分都正常运行
                    # self.player.play_unblock(w, self.wakeup_event)
                    self.player.play(w)

                # 如果在播音时收到唤醒词，应当先从play的播放循环中跳出，然后来到这里，跳出互动阶段
                if self.wakeup_event.is_set():
                    self.wakeup_event.clear()
                    break

            self.state = 0  # waiting
            self.player_exit_event.set()
            self.is_playing = False

class Package(object):

    @staticmethod
    def real_time_detection_result(img_data):
        r"""本数据包包含视觉模块的检测结果"""
        assert type(img_data) == str
        return {
            'type': 0,
            'data': img_data
        }

    @staticmethod
    def real_time_state(state):
        r"""本数据包是对前端心跳包请求的回应，回复后端当前的状态"""
        assert type(state) == int
        return {
            'type': 1,
            'data': state
        }

    @staticmethod
    def voice_to_word_result(data, success=True):
        r"""本数据包包含用户输入的语音转成文字的结果"""
        # if success: data may be the sentence of speaking. else: data may be the error message e.g. "network is error"
        assert type(success) == bool
        return {
            'type': 2,
            'success': success,
            'data': data
        }

    @staticmethod
    def response_word_result(data, success=True):
        r"""本数据包包含智能系统回复的文本结果"""
        # if success: data may be the sentence of response. else: data may be the error message e.g. "network is error"
        assert type(success) == bool
        return {
            'type': 3,
            'success': success,
            'data': data
        }


class ReceivePackage(object):
    def __init__(self, sock, addr, main_process):
        self.sock = sock
        self.addr = addr
        self.main_process = main_process
        return

    def __call__(self, data):
        try:
            data = json.loads(data)
            assert 'type' in data
        except:
            return False
        if data['type'] == 0:
            return self.start_speech(data)
        elif data['type'] == 1:
            return self.close_speech(data)
        elif data['type'] == 2:
            return self.exit_reading(data)
        elif data['type'] == 3:
            return self.heart_beat(data)
        else:
            return False

    def start_speech(self, *args):
        self.main_process.start_speech()
        return True

    def close_speech(self, *args):
        self.main_process.close_speech()
        return True

    def exit_reading(self, *args):
        self.main_process.close_reading()
        return True

    def heart_beat(self, data):
        if 'time' not in data or type(data['time']) != float:
            return False
        if str(self.addr) in clients:
            if clients[str(self.addr)]['last_active_time'] < float(data['time']):
                clients[str(self.addr)]['last_active_time'] = float(data['time'])
        else:
            clients[str(self.addr)] = {
                'sock': self.sock,
                'addr': self.addr,
                'last_active_time': float(data['time'])
            }
        return True


def sendPackage(sock, data):
    assert type(data) == dict
    data = json.dumps(data).strip() + '\n'
    try:
        sock.send(data.encode('utf-8'))
    except:
        pass
    return


def groupSendPackage(data):
    r"""将数据包群发给每一个前端"""
    global clients
    activate_clients = {}
    for addr in clients:
        if time.time() - clients[addr]['last_active_time'] > EXPIRE_TIME:
            print(addr, 'disconnected')
            clients[addr]['sock'].close()
        else:
            sendPackage(clients[addr]['sock'], data)
            activate_clients[addr] = clients[addr]
    clients = activate_clients
    return


def receivePackage(sock):
    history_data = b''
    while True:
        data = sock.recv(1)
        if data == b'':
            break
        if data == b'\n':
            yield history_data
            history_data = b''
        else:
            history_data += data
    return


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


def client_service(sock, addr, main_process):
    clients[str(addr)] = {
        'sock': sock,
        'addr': addr,
        'last_active_time': time.time()
    }
    # send initialization state to connecting socket.
    sendPackage(sock, Package.real_time_state(main_process.state))
    sendPackage(sock, Package.real_time_detection_result(main_process.detection_result))

    rp = ReceivePackage(sock, addr, main_process)
    for message in receivePackage(sock):
        rp(message)
    return


if __name__ == '__main__':
    main()
