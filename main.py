# -*- coding: utf-8 -*-
import os
import sys
import logging

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import cv2  # NOTE: 由于未知原因，先import cv2再import pyaudio会导致相机启动时报k4aException，所以这两行有先后顺序要求
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 把tensorflow的日志等级降低，不然输出一堆乱七八糟的东西
import tensorflow as tf
from matplotlib import cm
import time

import classifier
from audiohandler import Listener, Recorder, Player
from utils.utils import get_response, bytes_to_wav_data  # , TEST_INFO
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
RATE = 16000
LISTEN_SECONDS = 1
TOPK = 1
EXPECTED_WORD = "miya"

W_SMOOTH = 5
W_MAX = 10

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
    recorder = Recorder(FORMAT, CHANNELS, RATE, RECORDER_CHUNK_LENGTH)
    player = Player()
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

            img = cv2.imread(perception.savepath)
            for attr in result["attribute"]:
                # putText参数：np.ndarray, 文本左下角坐标(x, y), 字体, 文字缩放比例, (R, G, B), 厚度(不是高度)
                cv2.putText(img, attr["category"], attr["bbox"][:2], cv2.FONT_HERSHEY_COMPLEX, 0.6, BBOX_COLOR_DICT[attr["category"]],
                            thickness=2)
                cv2.rectangle(img, attr["bbox"][:2], attr["bbox"][2:], BBOX_COLOR_DICT[attr["category"]], thickness=2)
            cv2.imwrite("tmp2.jpg", img)

            recognized_str, response_list, wav_list = get_response(wav_data, result["attribute"])
            logger.info("Recognize result: " + recognized_str)

            # haven't said anything but pass VAD.

            if len(recognized_str) == 0 or "没事了" in recognized_str:
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

    classifier.load_graph("./models/CRNN_mia2.pb")
    labels = classifier.load_labels("./models/CRNN_mia2_labels.txt")

    listener = Listener(FORMAT, CHANNELS, RATE, LISTENER_CHUNK_LENGTH, LISTEN_SECONDS)
    player = Player()
    vpr = VoicePrint()

    with tf.Session() as sess:
        # main loop
        while True:
            history_probabilities = [np.zeros(len(labels)) for _ in range(W_SMOOTH)]
            smooth_probabilities = [np.zeros(len(labels)) for _ in range(W_MAX)]
            confidence = 0.0
            smooth_pred = ""

            listener.listen()
            # keyword spotting loop
            print("Listening...")
            while not (smooth_pred == EXPECTED_WORD and confidence > 0.5):
                frames = listener.buffer[-int(RATE / LISTENER_CHUNK_LENGTH * LISTEN_SECONDS):]
                wav_data = bytes_to_wav_data(b"".join(frames), FORMAT, CHANNELS, RATE)

                softmax_tensor = sess.graph.get_tensor_by_name("labels_softmax:0")
                mfcc_tensor = sess.graph.get_tensor_by_name("Mfcc:0")
                (predictions,), (mfcc,) = sess.run([softmax_tensor, mfcc_tensor], {"wav_data:0": wav_data})

                history_probabilities.pop(0)
                history_probabilities.append(predictions)

                smooth_predictions = np.sum(history_probabilities, axis=0) / W_SMOOTH

                # top_k = smooth_predictions.argsort()[-TOPK:][::-1]
                # for node_id in top_k:
                #     human_string = labels[node_id]
                #     score = smooth_predictions[node_id]
                #     logging.info('%s (score = %.5f)' % (human_string, score))

                pred_index = predictions.argsort()[-1:][::-1][0]
                pred = labels[pred_index]
                pred_score = predictions[pred_index]

                smooth_index = smooth_predictions.argsort()[-1:][::-1][0]
                smooth_pred = labels[smooth_index]
                smooth_score = smooth_predictions[smooth_index]

                smooth_probabilities.pop(0)
                smooth_probabilities.append(smooth_predictions)

                confidence = (np.prod(np.max(smooth_probabilities, axis=1))) ** (1 / len(labels))

                signals = np.frombuffer(b''.join(frames), dtype=np.int16)

                if PLOT:
                    plt.ion()
                    plt.subplot(221)
                    plt.title("Wave")
                    plt.ylim([-500, 500])
                    plt.plot(signals)
                    plt.subplot(222)
                    plt.title("Spectrogram")
                    plt.specgram(signals, NFFT=480, Fs=16000)
                    plt.subplot(223)
                    plt.title("MFCC")
                    plt.imshow(np.swapaxes(mfcc, 0, 1), interpolation='nearest', cmap=cm.coolwarm, origin='lower')
                    plt.subplot(224)
                    plt.axis("off")
                    plt.text(0, 0.5, 'predict: %s (score = %.5f)\nsmooth: %s (score = %.5f)\nconfidence = %.5f' % (
                        pred, pred_score, smooth_pred, smooth_score, confidence), ha="left", va="center")
                    plt.pause(0.1)
                    plt.clf()
                else:
                    # logger.info('predict: %s (score = %.5f)  smooth: %s (score = %.5f)  confidence = %.5f' % (
                    #     pred, pred_score, smooth_pred, smooth_score, confidence))
                    pass  # debug
            perception.send_single_image()
            haddata.value = False  # False要求重新向视觉模块获取视觉信息
            listener.stop()
            print("WAKEUP!")
            spk_name = vpr.get_spk_name(wav_data)

            # wakeup
            if not is_playing.value:
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


if __name__ == '__main__':
    main()
