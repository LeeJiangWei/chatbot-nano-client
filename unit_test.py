# 模块单独的测试放各自的文件里头，多个模块联动的可以考虑放这里
# 目的是减少每次调试调动的模块数，加快调试速度，降低调试难度
# -*- coding: utf-8 -*-
import os
import sys
import logging
import threading
import time
import json
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
        "你好米娅",
        # "今天天气怎么样",
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
        wav_data = str_to_wav_bin(s, speed=0.0)
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


def test_listener_with_tts_and_asr():
    r"""借助TTS和ASR模块测试listener的切片是否正确以及唤醒模型的输出的某些功能
    用tts模块生成语音，经过listener截取后，观察listener每个用来VAD的片段经过asr后输出结果是否符合预期，
    同时观察
    """
    print("Listener模块切片正确性检验开始...")

    input_list = [
        "今天天气怎么样",
        "你好米娅",
        "杯子在哪里",
        "咖啡机在哪里",
        "盆栽在哪里",
        "广州今天多云，气温16到24度~",
        "广州今天多云，气温16到24度。",
    ]
    
    # 因为这里用的是TTS的输出作为输入，所以输入采样率应该等于TTS输出的采样率
    input_rate = OUTPUT_RATE
    chunk_length = 30 * input_rate // 1000
    listener = Listener(FORMAT, CHANNELS, input_rate, chunk_length, LISTEN_SECONDS)
    asr = ASRVoiceAI(OUTPUT_RATE)
    
    for s in input_list:
        t1 = time.time()
        wav_data = str_to_wav_bin(s)
        listener.load_wav_data(wav_data)
        while listener.step_a_chunk():
            sliced_data = b"".join(listener.buffer[-int(input_rate / LISTENER_CHUNK_LENGTH * LISTEN_SECONDS):])
            result = asr.trans(sliced_data)
            print(f"{s} {result} 用时{time.time() - t1:.2f}s")
    print("Listener模块切片正确性检验结束。")


def test_listener_and_waker():
    r"""借助TTS模块，测试经Listener处理后的语音片段在Waker里的识别效果，顺便用Player模块
    目前还没有办法通过"你好米娅"的切片语音让Waker给出被唤醒的判断
    """
    print("Waker功能测试开始...")

    input_list = [
        "你好米娅",
        # "今天天气怎么样",
        # "杯子在哪里",
        # "咖啡机在哪里",
        # "盆栽在哪里",
        # "广州今天多云，气温16到24度~",
        # "广州今天多云，气温16到24度。",
    ]
    
    # 因为这里用的是TTS的输出作为输入，所以输入采样率应该等于TTS输出的采样率
    input_rate = OUTPUT_RATE
    listener = Listener(FORMAT, CHANNELS, input_rate, LISTENER_CHUNK_LENGTH, LISTEN_SECONDS)
    asr = ASRVoiceAI(OUTPUT_RATE)
    waker = Waker(EXPECTED_WORD)
    player = Player(rate=input_rate)

    for speed in range(0, 10):
        with tf.compat.v1.Session() as sess:
            for s in input_list:
                t1 = time.time()
                wav_data = str_to_wav_bin(s, speed=float(speed))  # 最低速度
                listener.load_wav_data(wav_data)
                while listener.step_a_chunk():
                    sliced_data = b"".join(listener.buffer[-int(input_rate / LISTENER_CHUNK_LENGTH * LISTEN_SECONDS):])
                    print(len(sliced_data))
                    result = asr.trans(sliced_data)
                    sliced_data = bytes_to_wav_data(sliced_data, FORMAT, CHANNELS, input_rate)
                    player.play(sliced_data)
                    waker.update(sliced_data, sess)
                    print(f"{s} {result} {waker.smooth_pred} {waker.confidence:.2f} {waker.waked_up()}  用时{time.time() - t1:.2f}s")
    print("Waker功能测试结束。")


def test_recorder_and_asr():
    r"""结合Recorder模块和ASR模块，测试固定时长阻塞式录制翻译和流式录制翻译，流式录制翻译预期应在录音结束后很短时间内
    （比如0.5s）就能给出翻译结果，且该时间不受录音时间长度影响
    """
    print("Recorder模块与ASR模块联合测试开始...")
    recorder = Recorder(rate=INPUT_RATE)
    asr = ASRVoiceAI(INPUT_RATE)

    record_seconds = 5  # 录制5s

    # 阻塞式录制+流式翻译=阻塞式录制翻译
    buffer = recorder.record(record_seconds)
    t1 = time.time()
    result = asr.trans(b"".join(buffer))
    print(f"{result}  非流式翻译用时{time.time() - t1:.2f}s")

    # 流式录制+流式翻译=流式录制翻译
    finish_record_event = threading.Event()
    record_proc = threading.Thread(target=recorder.record,
                         kwargs=({
                             "record_seconds": record_seconds,
                             "finish_record_event": finish_record_event
                         }))
    record_proc.setDaemon(True)
    record_proc.start()

    t1 = time.time()
    result = asr.trans_stream(recorder.buffer, finish_record_event)
    print(f"{result}  录制用时{record_seconds}s, 流式翻译用时{time.time() - t1:.2f}s")
    print("Recorder模块与ASR模块联合测试结束。")


def test_recorder_auto_and_asr():
    r"""结合Recorder模块和ASR模块，测试自动捕捉阻塞式录制翻译和流式录制翻译，流式录制翻译预期应在录音结束后很短时间内
    （比如0.5s）就能给出翻译结果，且该时间不受录音时间长度影响
    """
    print("Recorder模块与ASR模块联合测试开始...")
    recorder = Recorder(rate=INPUT_RATE, channels=1)
    # NOTE: ASRVoiceAI不需要设置通道数吗？
    asr = ASRVoiceAI(INPUT_RATE)

    # 阻塞式录制+流式翻译=阻塞式录制翻译
    print("Recording...")
    t1 = time.time()
    buffer, _ = recorder.record_auto()
    result = asr.trans(b"".join(buffer))
    print(f"{result}  非流式录制翻译用时{time.time() - t1:.2f}s")

    # 流式录制+流式翻译=流式录制翻译
    print("Recording...")
    t1 = time.time()
    finish_record_event = threading.Event()
    record_proc = threading.Thread(target=recorder.record_auto,
                         kwargs=({
                             "finish_record_event": finish_record_event
                         }))
    record_proc.setDaemon(True)
    record_proc.start()

    result = asr.trans_stream(recorder.buffer, finish_record_event)
    print(f"{result}  流式录制翻译用时{time.time() - t1:.2f}s")
    print("Recorder模块与ASR模块联合测试结束。")


def test_interact():
    r"""测试主程序中的interact函数，主要是通过大量用例测试rasa的输出结果是否正常"""
    from main import MainProcess,client_service
    import socket

    main_process = MainProcess()
    main_process.init_backend()

    sock = socket.socket()
    # 该配置允许后端在断开连接后端口立即可用，也即没有TIME_WAIT阶段，调试必备
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(SOCKET)
    sock.listen(5)
    recv_sock, recv_addr = sock.accept()
    t = threading.Thread(target=client_service, args=(recv_sock, recv_addr, main_process, ))
    t.setDaemon(True)
    t.start()

    # 目前的键值有：weather location chat want position color quantity function
    with open("rasa_querys.json", "r") as fp:
        data = json.load(fp)

    results = []

    for key, query_texts in data.items():
        print(f"测试rasa对{key}类问题的回答..")
        for query_text in query_texts:
            answer = main_process.interact(query_text)
            # 在中英文混合的字符串s中，len(s)表示s所包含的字符数，中文英文都算1个字符，而中文在print的时候实际上是
            # 占了两个英文字符的宽度。len(s.encode("gbk")) - len(s) 可以得到s中中文字符的数量，因为编码成gbk之后，
            # 一个英文字符长度为1，一个中文字符长度为2，所以多出来的长度刚好就是中文字符数量。字符串的ljust方法补足的
            # 是字符数，也即len(s)，因此直接s.ljust(30)是不行的，含有中文字符多的字符串，最后print出来就会更胖，
            # 因此需要做一些处理：如果是中文符号，那么要补至的长度就减1，因为这个中文符号最后一个顶俩，print出来
            # 的长度又加1加了回去
            results.append(f"{query_text + ' ' * (25 - len(query_text.encode('gbk')))} --> {answer}\n")

    with open("rasa_output.txt", "w") as fp:
        fp.writelines(results)


if __name__ == "__main__":
    # test_tts_and_player()
    # test_tts_and_asr()
    # test_cam_and_qas()
    # test_player_ws()
    # test_listener_with_tts_and_asr()
    # test_listener_and_waker()
    # test_recorder_and_asr()
    # test_recorder_auto_and_asr()
    test_interact()
