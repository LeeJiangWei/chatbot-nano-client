import io
import wave

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import tensorflow as tf
from matplotlib import cm

import classifier
from audiohandler import ListenThread, Recorder

CHUNK_LENGTH = 1000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
LISTEN_SECONDS = 1
TOPK = 1

W_SMOOTH = 5
W_MAX = 10

if __name__ == '__main__':
    classifier.load_graph("./models/CRNN/CRNN_L.pb")
    labels = classifier.load_labels("./models/labels.txt")

    history_probabilities = [np.zeros(len(labels)) for _ in range(W_SMOOTH)]
    smooth_probabilities = [np.zeros(len(labels)) for _ in range(W_MAX)]
    confidence = 0.0

    listener = ListenThread(FORMAT, CHANNELS, RATE, CHUNK_LENGTH, LISTEN_SECONDS)
    listener.start()

    plt.ion()

    with tf.Session() as sess:
        while True:
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
            #     print('%s (score = %.5f)' % (human_string, score))

            pred_index = predictions.argsort()[-1:][::-1][0]
            pred = labels[pred_index]
            pred_score = predictions[pred_index]

            smooth_index = smooth_predictions.argsort()[-1:][::-1][0]
            smooth = labels[smooth_index]
            smooth_score = smooth_predictions[smooth_index]

            smooth_probabilities.pop(0)
            smooth_probabilities.append(smooth_predictions)

            confidence = (np.prod(np.max(smooth_probabilities, axis=1))) ** (1 / len(labels))
            # print('confidence = %.5f' % confidence)

            signals = np.frombuffer(b''.join(frames), dtype=np.int16)

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
                pred, pred_score, smooth, smooth_score, confidence), ha="left", va="center")
            plt.pause(0.1)
            plt.clf()
