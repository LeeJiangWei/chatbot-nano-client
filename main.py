# -*- coding: utf-8 -*-
import os
import sys
import logging
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
from utils.vision_utils import get_color_dict
from api import VoicePrint, str_to_wav_bin
from vision_perception import VisionPerception
from vision_perception.client_for_voice import InfoObtainer
import multiprocessing as mp

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


def interact_process(wakeup_event, is_playing, player_exit_event, all_exit_event, haddata):
    recorder = Recorder(FORMAT, CHANNELS, INPUT_RATE, RECORDER_CHUNK_LENGTH)
    player = Player(rate=OUTPUT_RATE)
    while True:
        print("Wait to be wakeup...")
        wakeup_event.wait()
        wakeup_event.clear()
        while True:
            is_playing.value = True
            logger.info("Start recording...")
            wav_list, no_sound = recorder.record_auto()
            logger.info("Stop recording.")
            if no_sound:
                logger.info("No sound detected, conversation canceled.")
                break

            if wakeup_event.is_set():
                wakeup_event.clear()
                break

            wav_data = bytes_to_wav_data(b"".join(wav_list))
            logger.info("Waiting server...")

            # perception.send_single_image()
            # haddata.value = False  # False要求重新向视觉模块获取视觉信息
            # time.sleep(0.1)  # 给视觉模块时间重置信息
            data = {"require": "attribute"}
            result = I.obtain(data, haddata.value)
            haddata.value = True  # 获得过一次视觉信息之后，在下次重新唤醒之前就不需要重复获取了
            
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

            recognized_str, response_list, wav_list = get_response(wav_data, result["attribute"])
            logger.info("Recognize result: " + recognized_str)

            # haven't said anything but pass VAD.
            if len(recognized_str) == 0 or "没事" in recognized_str:
                break

            # TODO: 把退出主程序的功能做好，现在没有实现预期功能
            if recognized_str in ["退出。", "关机。"]:
                print("退出互动环节...")
                all_exit_event.set()
                time.sleep(10)  # 坐等主进程退出
                break

            if wakeup_event.is_set():
                wakeup_event.clear()
                break

            for r, w in zip(response_list, wav_list):
                logger.info(r)
                # save_wav(w, "tmp.wav", rate=player.rate)  # debug临时语句，保存原本音频流以确保play之前的部分都正常运行
                player.play_unblock(w, wakeup_event)

            # interrupt
            if wakeup_event.is_set():
                wakeup_event.clear()
                break

        player_exit_event.set()
        is_playing.value = False


def main():
    r"""
    时序图：─│┌┐└┘├┴┬┤┼╵
                                                      ┌─没检测到唤醒词─────────────────────────────────────────────┐
    初始阶段：main() ─── listener.listen()录制用户语音 ─┴─检测到唤醒词── 播放欢迎语 ─── wakeup_event.set() ─── 继续listener.listen()
                                                                                │ 主进程可以在record_auto()结束时、播放语音前或者播放语音时通过说唤醒词退出互动阶段
    互动阶段：wakeup_event.set() ─── 进入interact_process循环 ─── recorder.record_auto()录制用户语音 ─── 向视觉模块获取信息 ─── 结合视觉信息向语音模块获取回复的音频数据 ─── player.play_unblock()播放语音回复 ─── wakeup_event.clear()
    """
    wakeup_event = mp.Event()
    is_playing = mp.Value('i', 0)
    player_exit_event = mp.Event()
    all_exit_event = mp.Event()
    haddata = mp.Value('i', False)  # 用于InfoObtainer的共享变量

    inter_proc = mp.Process(target=interact_process, args=(wakeup_event, is_playing, player_exit_event, all_exit_event, haddata))
    inter_proc.daemon = True  # 设置成守护进程，不然ctrl+C退出main()子进程还在，程序依然卡死
    inter_proc.start()

    listener = Listener(FORMAT, CHANNELS, INPUT_RATE, LISTENER_CHUNK_LENGTH, LISTEN_SECONDS)
    player = Player(rate=OUTPUT_RATE)
    vpr = VoicePrint()
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
            haddata.value = False  # False要求重新向视觉模块获取视觉信息
            listener.stop()
            print("WAKEUP!")
            # save_wav(b"".join(frames), "tmp.wav")  # debug临时语句，保存原本音频流以确保play之前的部分都正常运行

            # wakeup
            if not is_playing.value:
                t1 = time.time()
                spk_name = vpr.get_spk_name(wav_data)
                print(f"声纹识别耗费时间为：{time.time() - t1:.2f}秒")
                wav = str_to_wav_bin(spk_name + '你好!')
            # interrupt
            elif is_playing.value:
                wakeup_event.set()
                wav = str_to_wav_bin("我在!")
                player_exit_event.wait()

            player_exit_event.clear()
            # 这里的play只负责播放欢迎语
            player.play(wav)
            wakeup_event.set()

            if all_exit_event.is_set():  # 正常退出，不使用ctrl+C退出，不然相机老是出幺蛾子
                print("退出主程序。")
                break


if __name__ == "__main__":
    main()
