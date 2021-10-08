# -*- coding: utf-8 -*-
import io
import sys
import wave
import logging

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import tensorflow as tf
from matplotlib import cm
import time

import classifier
from audiohandler import Listener, Recorder, Player
from utils.utils import get_response#, TEST_INFO
from api import VoicePrint, str_to_wav_bin
from vision_perception import VisionPerception
from vision_perception.client_for_voice import InfoObtainer
from multiprocessing import Process, Value
import multiprocessing

HOST = '222.201.134.203'
PORT = 17000
PORT_INFO = 17001
perception = VisionPerception(HOST, PORT)
I = InfoObtainer(HOST, PORT_INFO)

RECORDER_CHUNK_LENGTH = 30
CHUNK_LENGTH = 1000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
LISTEN_SECONDS = 1
TOPK = 1
EXPECTED_WORD = "miya"

W_SMOOTH = 5
W_MAX = 10

PLOT = False

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                              datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)


def interact_process(wakeup_event, is_playing, player_exit_event):
    recorder = Recorder(FORMAT, CHANNELS, RATE, RECORDER_CHUNK_LENGTH)
    player = Player()
    while True:
        print("Wait to be wakeup...")
        wakeup_event.wait()
        wakeup_event.clear()
        while True:
            is_playing.value = True
            logger.info("Start recording...")
            wav, _flags = recorder.record_auto()
            logger.info("Stop recording.")
            if not _flags:
                logger.info("No sound detected, conversation canceled.")
                break

            if wakeup_event.is_set():
                break

            container = io.BytesIO()
            wf = wave.open(container, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(wav))
            wf.close()
            container.seek(0)
            wav_data = container.read()
            logger.info("Waiting server...")
            data = {"require": "attribute"}
            result = I.obtain(data)
            from pprint import pprint
            pprint(result["attribute"])
            print("时间戳:", result["timestamp"])
            # img = cv2.imread(perception.savepath)
            # for attr in result["attribute"]:
            #     # putText参数：np.ndarray, 文本左下角坐标(x, y), 字体, 文字缩放比例, (R, G, B), 厚度(不是高度)
            #     cv2.putText(img, attr["category"], attr["bbox"][:2], cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), thickness=2)
            #     cv2.rectangle(img, attr["bbox"][:2], attr["bbox"][2:], (0,255,0), thickness=2)
            # cv2.imwrite("tmp2.jpg", img)            

            recognized_str, response_list, wav_list = get_response(wav_data, result["attribute"])
            logger.info("Recognize result: " + recognized_str)

            # haven't said anything but pass VAD.

            if len(recognized_str) == 0 or "没事了" in recognized_str:
                break

            if wakeup_event.is_set():
                break


            for r, w in zip(response_list, wav_list):
                logger.info(r)
                player.play_unblock(w, wakeup_event)
                print("exit")

            # interrupt
            if wakeup_event.is_set():
                break

        player_exit_event.set()
        is_playing.value = False


def main():
    wakeup_event = multiprocessing.Event()
    is_playing= multiprocessing.Value('i',0)
    player_exit_event = multiprocessing.Event()

    inter_proc = Process(target=interact_process, args=(wakeup_event, is_playing, player_exit_event))
    inter_proc.start()

    classifier.load_graph("./models/CRNN_mia2.pb")
    labels = classifier.load_labels("./models/CRNN_mia2_labels.txt")

    listener = Listener(FORMAT, CHANNELS, RATE, CHUNK_LENGTH, LISTEN_SECONDS)
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
                frames = listener.buffer[:int(RATE / CHUNK_LENGTH * LISTEN_SECONDS)]

                container = io.BytesIO()
                wf = wave.open(container, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                container.seek(0)
                wav_data = container.read()

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
            I.reset()
            listener.stop()
            print("WAKEUP!")
            spk_name = vpr.get_spk_name(wav_data)

            wakeup_event.set()
            # wakeup
            if not is_playing.value:
                wav = str_to_wav_bin(spk_name + '你好!')
            # interrupt
            elif is_playing.value:
                wav = str_to_wav_bin("我在!")
                player_exit_event.wait()

            player_exit_event.clear()
            player.play(wav)


if __name__ == '__main__':
    main()
