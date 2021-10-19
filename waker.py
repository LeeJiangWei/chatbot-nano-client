import os
import sys
import logging
import time
import wave
from matplotlib import cm
from matplotlib import pyplot as plt

import pyaudio
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 把tensorflow的日志等级降低，不然输出一堆乱七八糟的东西
import tensorflow as tf

from audiohandler import Listener, Player
from utils.utils import bytes_to_wav_data, save_wav

# Tensorboard visualize command:
# python -m tensorflow.python.tools.import_pb_to_tensorboard --model_dir="./models/CRNN/CRNN_L.pb"  --log_dir="./tmp"
# tensorboard --logdir=./tmp

LISTENER_CHUNK_LENGTH = 1000  # 一个块=1s的语音
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
LISTEN_SECONDS = 1
EXPECTED_WORD = "miya"

PLOT = False

def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.io.gfile.GFile(filename)]


def load_graph(filename):
    """Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def run_graph(wav_data, labels, num_top_predictions):
    """Runs the audio data through the graph and prints predictions."""
    with tf.Session() as sess:
        # Feed the audio data as input to the graph.
        #   predictions  will contain a two-dimensional array, where one
        #   dimension represents the input image count, and the other has
        #   predictions per class
        softmax_tensor = sess.graph.get_tensor_by_name("labels_softmax:0")
        predictions, = sess.run(softmax_tensor, {"wav_data:0": wav_data})

        # Sort to show labels in order of confidence
        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

        return 0


W_SMOOTH = 5
W_MAX = 10
TOPK = 1
class Waker:
    def __init__(self, wakeup_word, threshold=0.5, w_smooth=W_SMOOTH, w_max=W_MAX):
        self.wakeup_word = wakeup_word
        self.threshold = threshold
        self.w_smooth = w_smooth
        self.w_max = w_max
        
        load_graph("./models/CRNN_mia2.pb")
        self.labels = load_labels("./models/CRNN_mia2_labels.txt")
        self.history_probabilities = [np.zeros(len(self.labels)) for _ in range(self.w_smooth)]
        self.smooth_probabilities = [np.zeros(len(self.labels)) for _ in range(self.w_max)]
        self.confidence = 0.0
        self.smooth_pred = ""

    def waked_up(self):
        # print(self.smooth_pred, self.confidence)
        return (self.smooth_pred == self.wakeup_word and self.confidence > self.threshold)

    def update(self, wav_data, sess, plot=False):
        # NOTE: 不可以把with tf.compat.v1.Session() as sess:放在这里，开关session的开销非常大
        # 放在这里会导致函数执行时间变为原来的3倍
        softmax_tensor = sess.graph.get_tensor_by_name("labels_softmax:0")
        mfcc_tensor = sess.graph.get_tensor_by_name("Mfcc:0")
        (predictions,), (mfcc,) = sess.run([softmax_tensor, mfcc_tensor], {"wav_data:0": wav_data})

        self.history_probabilities.pop(0)
        self.history_probabilities.append(predictions)

        smooth_predictions = np.sum(self.history_probabilities, axis=0) / W_SMOOTH

        # top_k = smooth_predictions.argsort()[-TOPK:][::-1]
        # for node_id in top_k:
        #     human_string = labels[node_id]
        #     score = smooth_predictions[node_id]
        #     logging.info('%s (score = %.5f)' % (human_string, score))

        pred_index = predictions.argsort()[-1:][::-1][0]
        pred = self.labels[pred_index]
        pred_score = predictions[pred_index]

        smooth_index = smooth_predictions.argsort()[-1:][::-1][0]
        self.smooth_pred = self.labels[smooth_index]
        smooth_score = smooth_predictions[smooth_index]

        self.smooth_probabilities.pop(0)
        self.smooth_probabilities.append(smooth_predictions)

        self.confidence = (np.prod(np.max(self.smooth_probabilities, axis=1))) ** (1 / len(self.labels))

        # wav_data含有44个字节的wav格式头，后面的才是音频数据
        signals = np.frombuffer(wav_data[44:], dtype=np.int16)

        if plot:  # 仅用于观察噪声波形，正式使用时不可以plot，因为目前版本外面的listener.listen()跟这里是串行的，plt.pause()的时间内场景的声音将不被录制
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
                pred, pred_score, self.smooth_pred, smooth_score, self.confidence), ha="left", va="center")
            plt.pause(0.1)
            plt.clf()
        else:
            # logger.info('predict: %s (score = %.5f)  smooth: %s (score = %.5f)  confidence = %.5f' % (
            #     pred, pred_score, smooth_pred, smooth_score, confidence))
            pass  # debug
            
    def reset(self):
        self.history_probabilities = [np.zeros(len(self.labels)) for _ in range(self.w_smooth)]
        self.smooth_probabilities = [np.zeros(len(self.labels)) for _ in range(self.w_max)]
        self.confidence = 0.0
        self.smooth_pred = ""


def test_waker():
    r"""对Waker的功能进行基本测试，需要借助Listener"""
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                datefmt='%Y/%m/%d %H:%M:%S')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    listener = Listener(FORMAT, CHANNELS, RATE, LISTENER_CHUNK_LENGTH, LISTEN_SECONDS)
    waker = Waker(EXPECTED_WORD)

    with tf.compat.v1.Session() as sess:
        # main loop
        while True:
            listener.listen()
            # keyword spotting loop
            print("Listening...")
            while not waker.waked_up():
                # frames包含了最近LISTEN_SECONDS内的音频数据
                frames = listener.buffer[-int(RATE / LISTENER_CHUNK_LENGTH * LISTEN_SECONDS):]
                wav_data = bytes_to_wav_data(b"".join(frames), FORMAT, CHANNELS, RATE)
                waker.update(wav_data, sess, PLOT)
                print(waker.smooth_pred, waker.confidence)

            # 此时wav_data即为使waker唤醒的最近LISTEN_SECONDS秒的音频
            save_wav(wav_data, "tmp.wav", FORMAT, CHANNELS, RATE)
            waker.reset()  # 重置waker的置信度等参数，使其下轮循环能重新进入内层while循环，等待下一次唤醒
            listener.stop()
            print("WAKEUP!")
            time.sleep(1.5)


def test_waker_using_wav():
    r"""使用事先录制好的wav文件让Waker进行判断，预期原本能唤醒的句子，录下来之后喂数据给Waker也应当能唤醒
    注意模型只支持16K采样率1s长音频，其他格式能放进去，但是不能保证效果
    """
    waker = Waker("miya")
    # 借助Player播放一下片段
    player = Player(rate=RATE)

    # 块长度选0.2s，也就是步长0.2s，宽度为1s的滑动窗口
    chunk_length = int(0.2 * RATE * CHANNELS * 2) # 单声道x1 Int16x2
    
    with tf.compat.v1.Session() as sess:
        for i in range(1, 6):
            with wave.open(f"test{i}.wav", "rb") as wf:
                wav_data = wf.readframes(wf.getnframes())
            seek = 0
            while seek + RATE <= len(wav_data):
                data = wav_data[seek: seek + RATE]
                player.play_unblock(data)
                data = bytes_to_wav_data(data, FORMAT, CHANNELS, RATE)
                waker.update(data, sess)
                print(waker.smooth_pred, waker.confidence, waker.waked_up())
                seek += chunk_length


if __name__ == "__main__":
    # test_waker()
    test_waker_using_wav()
