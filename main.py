import io
import sys
import wave
import logging

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import tensorflow as tf
from matplotlib import cm

import classifier
from audiohandler import Listener, Recorder, Player
from utils.utils import get_response, TEST_INFO
from api import VoicePrint,str_to_wav_bin
from vision_perception import VisionPerception


HOST = '222.201.134.203'
PORT = 17000
perception = VisionPerception(HOST, PORT)

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


def main():
    classifier.load_graph("./models/CRNN_mia2.pb")
    labels = classifier.load_labels("./models/CRNN_mia2_labels.txt")

    listener = Listener(FORMAT, CHANNELS, RATE, CHUNK_LENGTH, LISTEN_SECONDS)
    recorder = Recorder(FORMAT, CHANNELS, RATE, RECORDER_CHUNK_LENGTH)
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
                    logger.info('predict: %s (score = %.5f)  smooth: %s (score = %.5f)  confidence = %.5f' % (
                        pred, pred_score, smooth_pred, smooth_score, confidence))
            perception.send_single_image()
            listener.stop()

            spk_name=vpr.get_spk_name(wav_data)
            wav=str_to_wav_bin(spk_name+'你好!')

            player.play(wav)

            # interact loop
            while True:
                logger.info("Start recording...")
                wav, _flags = recorder.record_auto()
                logger.info("Stop recording.")

                if not _flags:
                    logger.info("No sound detected, conversation canceled.")
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
                recognized_str, response_list, wav_list = get_response(wav_data, TEST_INFO)
                logger.info("Recognize result: " + recognized_str)
                for r, w in zip(response_list, wav_list):
                    logger.info(r)
                    player.play(w)


if __name__ == '__main__':
    main()
