# 模块单独的测试放各自的文件里头，多个模块联动的可以考虑放这里
# 目的是减少每次调试调动的模块数，加快调试速度，降低调试难度
# -*- coding: utf-8 -*-
import os
import sys
import logging
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
from utils.tts_utils import TTSBiaobei
from api import (str_to_wav_bin_unblock, str_to_wav_bin,
                 VoicePrint)
from vision_perception import K4aCamera, NormalCamera

from vision_perception.client_for_voice import InfoObtainer


HOST = "222.201.134.203"
PORT = 17000
PORT_INFO = 17001

RECORDER_CHUNK_LENGTH = 480  # 一个块=30ms的语音
LISTENER_CHUNK_LENGTH = 1000
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


def test_tts_and_player():
    r"""Author: zhang.haojian
    测试TTS模块的性能
    """
    print("TTS模块功能测试开始...")
    # 每个元素就是一个测试用例
    input_list = [
        # "今天天气怎么样",
        # "你好米娅",
        # "杯子在哪里",
        # "咖啡机在哪里",
        # "盆栽在哪里",
        "广州今天多云，气温16到24度~",
        # "广州~今天~多云~气温16到24度~",
        # "广州~今天~多云~，气温16到24度~",
        "广州今天多云，气温16到24度。",
        "广州今天多云，气温16到24度~。",  # NOTE: 通过该用例发现，TTS在以~或~。结尾时，转换的音频会出现戛然而止的情况！
        "广州今天多云，气温16到24度",
    ]

    # str_to_wav_bin的输出采样率是OUTPUT_RATE，所以这里是OUTPUT_RATE，不是INPUT_RATE
    player = Player(rate=OUTPUT_RATE)

    t1 = time.time()
    for s in input_list:
        wav_data = str_to_wav_bin(s)
        player.play_unblock(wav_data)
        print(f"{s}  用时{time.time() - t1:.2f}s")
        t1 = time.time()
    print("TTS模块功能测试结束。")


def test_tts_and_asr(input_list=[]):
    r"""Author: zhang.haojian
    测试TTS模块和ASR模块的性能，将一个文本输入TTS模块转成语音，再经过ASR模块转成文本，预期应与原始文本一致
    """
    print("TTS模块和ASR模块联合功能测试开始...")
    # 每个元素就是一个测试用例
    input_list = [
        "今天天气怎么样",
        "你好米娅",
        "杯子在哪里",
        "咖啡机在哪里",
        "盆栽在哪里",
    ]

    asr = ASRVoiceAI(OUTPUT_RATE)
    t1 = time.time()
    for s in input_list:
        wav_data = str_to_wav_bin(s)
        # str_to_wav_bin的输出采样率是OUTPUT_RATE，所以这里是OUTPUT_RATE，不是INPUT_RATE
        result = asr.trans(wav_data)
        print(f"{s} {result} {s == result}  用时{time.time() - t1:.2f}s")
        t1 = time.time()
    print("TTS模块和ASR模块联合功能测试结束。")


def test_cam_and_qas(input_list=[]):
    r"""Author: zhang.haojian
    测试摄像头跟问答系统(question answer system, qas)，在系统实际运行时它将收到ASR模块的输出
    Args:
        recognized_str_list (list): 每个元素是一个str类型的测试用例
    """
    print("摄像头跟问答系统联合功能测试开始...")
    # 每个元素就是一个测试用例
    input_list = [
        "今天天气怎么样",
        "广东广州",
        "你好米娅",
        # "没事了",  # 退出词，如果启用，预期应在这个位置退出循环
        "杯子在哪里",
        "咖啡机在哪里",
        "盆栽在哪里",
        # "",  # 无人声，如果启用，预期应在这个位置退出循环
        "讲一个笑话吧",
        "",  # 表示最后的静音段
    ]

    I = InfoObtainer(HOST, PORT_INFO)
    # NOTE: 要用到摄像头时记得加上sudo！
    scene_cam = K4aCamera(HOST, PORT)

    # 当前版本一次唤醒发送一张图片
    scene_cam.send_single_image()
    for recognized_str in input_list:
        t1 = time.time()
        logger.info("Waiting server...")

        data = {"require": "attribute"}
        result = I.obtain(data, False)
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
        t2 = time.time()

        print("近义词替换前：", recognized_str)
        recognized_str = synonym_substitution(recognized_str)
        print("近义词替换后：", recognized_str)

        if len(recognized_str) == 0:
            recognized_str = "(无人声)"
        if recognized_str == "(无人声)" or "没事" in recognized_str:
            break

        response_word, sentences = get_answer(recognized_str, result["attribute"])
        print(response_word, sentences)
        print(f"请求视觉信息用时: {t2 - t1:.2f}s, rasa用时: {time.time() - t2:.2f}s")
        recognized_str = recognized_str.replace("~", "")

    print("摄像头跟问答系统联合功能测试结束。")
    return


def test_player_ws(input_list=[]):
    r"""Author: zhang.haojian
    测试使用WebSocket (ws)的流式输出。流式输出的场景是，句子有很多，如果等到全部TTS完毕再开始播音，中间会有
    很长时间的静音，用户体验不好。
    """
    print("Player模块和TTS模块流式输出测试开始...")
    # 每个元素就是一个测试用例
    input_list = [
        "蜈蚣说：俺可发挥腿多的优势，有专练罚点球的，有专练踢弧线球的，还有专练带球过人的。万一俺的腿受了伤，根本不用下场，俺可用备用腿接着踢。",
        "学霸聚在一起，总是会研究考题的类型，和解决考题的办法。学渣聚在一起，永远都是研究如何抄袭，如何躲避老师的眼睛抄到更多的题。",
        "去医院体检，医生拿着我的报告单，说：“幸好你来得早啊！”在我惊出一身冷汗的时候，医生不慌不忙的说道：“再晚点，我就下班了。”",
        "昨天起来特别饿，就叫了肯德基。过半小时说到了。下楼看小伙翻开后面的箱子，一看什么都没有。正想问他，只听他默默说了句：“骑错车了。”",
        "瘦子有瘦子的宿命，胖子有胖子的宿命。瘦子就算饿到皱起了眉头，也仍然被当成是忧郁；胖子就算忧郁到皱起了眉头，也仍然被当成是饿了。",
        "孟姜女得知丈夫在修长城的时候累死了，尸骨埋在了长城里。她没有见到丈夫最后一面很是难过，拍着长城失声痛哭地大喊道：“挖掘技术哪家强？！”",
    ]

    player = Player(rate=OUTPUT_RATE)
    tts = TTSBiaobei()

    for response_word in input_list:
        start = time.time()
        ws = tts.start_tts(response_word)

        print(f"输出: {response_word}:", end="")
        player.play_unblock_ws(ws)
        print(f"  用时 {time.time() - start:.2f}s")

    print("Player模块和TTS模块流式输出测试结束。")


if __name__ == "__main__":
    # test_tts_and_player()
    # test_tts_and_asr()
    # test_cam_and_qas()
    test_player_ws()