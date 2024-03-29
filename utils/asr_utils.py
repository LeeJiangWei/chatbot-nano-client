# -*- coding: utf-8 -*-
# 改编自voiceai的transcribe_websocket.py
import time
import json
import websocket
import _thread
import ssl
import traceback
from datetime import datetime

t1 = datetime.now()

# on_开头的这些函数都会被websocket.WebSocketApp类实例调用，被当成成员函数，因此第一个参数是self并不是写错了。
# 我也不知道为啥要用这种写法，拿到的example代码就是这样写的。
# 然后调用它们的时候，ws这个变量在这些函数的外部，所以在函数内部可以直接调用，虽然看起来没有定义，但是实际运行起来是
# 不会报错的
def on_message_test(self, message):
    r"""
    Args:
        message (bytes): utf-8 data received from the server
    """

    text = json.loads(message, encoding="utf-8")

    if "isFinal_" not in text:
        return

    if text["isFinal_"] == True:
        t2 = datetime.now()
        print(t2)
        delta = t2 - t1
        elapsed_milliseconds = delta.microseconds / 1000 + delta.seconds * 1000
        print("Duration = {} milliseconds".format(elapsed_milliseconds))

    print(text["isFinal_"])
    print(text["alternatives_"][0]["transcript_"])


def on_message(self, message):
    r"""
    Args:
        message (bytes): utf-8 data received from the server
    """
    text = json.loads(message, encoding="utf-8")
    # 在"liveRecordingScene"配置为False的情况下，用户的一次对话只会提交一次请求，收到一次响应，因此就不需要判断isFinal_了
    if "isFinal_" not in text:  # 最开始建立websocket时发了一个配置，服务器会把配置发回来，此时没有isFinal_，也不是ASR的数据包，忽略之
        return
    self.asr_result = text["alternatives_"][0]["transcript_"]


def on_error(self, error):
    print("------------------------------- ERROR ", error)


def on_close(self, status_code, close_msg):
    r"""
    Args:
        status_code (int): 状态码，如1000, 1011
        close_msg (str): 返回的信息，貌似正常关闭时都是个空字符串
    """
    pass
    # print("------------------------------- CLOSE ")


def run(ws):
    r"""发送整句话，连续把数据发完，最后发送一个结束符"""
    block = 16000  # 一次截取16000字节
    start, seek = 0, block

    while start < len(ws.wav_data):
        audio = ws.wav_data[start: seek]
        ws.send(audio, websocket.ABNF.OPCODE_BINARY)
        start += block
        seek += block

    # 最后发送一个end of stream控制信令，通知Websocket Server音频流传输已经EOF，而后Server会主动断开连接
    endOfStreamEvent = { "endOfStream": True }
    ws.send(json.dumps(endOfStreamEvent))


def run_stream(ws):
    r"""有数据就一直发送，直到收到停止录制的信号时发送一个结束符，然后停下"""
    # 刚好有个buffer成员，直接一块一块发，舒服
    idx = 0
    while True:
        while idx < len(ws.buffer):
            # 待发送序号小于buffer长度时，持续发送数据
            audio = ws.buffer[idx]
            ws.send(audio, websocket.ABNF.OPCODE_BINARY)
            idx += 1

        if ws.finish_record_event.is_set():
            # 待发送序号大于等于buffer长度且音频录制已结束时，发送结束符，结束发送过程
            endOfStreamEvent = { "endOfStream": True }
            ws.send(json.dumps(endOfStreamEvent))
            break
        time.sleep(0.1)  # 没有数据可发但尚处于录制状态中，那就等一等


def on_open(self):
    config = {
        ###### 必须的参数 ######
        # 即将要发送的语音数据的采样率，数值为8000、16000、44100、48000等；
        # 注意：这个数值必须跟语音实际的采样率一致。
        "sampleRate": self.rate,
        # 是否在转写文本中自动加标点
        "addPunctuation": True,
        # 【注：机器⼈项目不需要打开这个功能】是否需要返回转写文本中每个标点的时间信息
        "needTimeinfo": False,
        # 是否在转写文本中自动将数字转换为阿拉伯数字
        "convertNumbers": True,
        # 静音检测时⻓，单位为10毫秒，大小为[50, 500]
        "pauseTime": 150,

        # 是否为Conference类似的实时转写场景，例如，实时连续转写的会议场景应用，需将这个字段设置为True。
        # 如果为True，则每当有（稳定的或不稳定的）转写文本产生，该转写文本将会马上被Server推送给Client；
        # 如果为False，则**每句**转写文本将会一直被Server缓存累积着，直到Client发送一个End Of Stream
        # 控制信令消息通知"语音数据流传输已经EOF，不会再有新数据发送"，此时Server会将累积的转写文本一次性
        # 推送给Client，同时Server不再继续接收新的语音数据流！
        # 不管发送几句话都可以用False，如果用True将会持续收到服务端发回的数据，我们现在不需要在录制的同时
        # 显示当前翻译的文本，直接最后一起返回就可以了，所以用False
        "liveRecordingScene": False,

        ###### 可选的参数 ######
        # 【注：机器⼈项目不需要打开这个功能】是否在转写文本中将口语转换为书面语
        "oral2written": False,
        # 如果没有协商`modelName`这个参数，则默认是让服务器选择使用`通用`模型。
        "modelName": "susie-lvcsr4-general-cn-16k"
    }

    self.send(json.dumps(config))  # 建立连接之后先发送配置，然后才能开始发语音

    _thread.start_new_thread(self.run, (self,))  # 启动线程执行run()函数发送数据


def on_open_test(self):

    config = {
        "sampleRate": 8000,
        "addPunctuation": True,
        "needTimeinfo": False,
        "convertNumbers": True,
        "pauseTime": 150,
        "liveRecordingScene": False,
        "oral2written": False,
        "modelName": "susie-lvcsr4-general-cn-16k"
    }
    self.send(json.dumps(config))  # 建立连接之后先发送配置，然后才能开始发语音

    def run_test():
        with open("../welcome.wav", "rb") as f:
            audio = f.read(16000)
            while audio: 
                self.send(audio, websocket.ABNF.OPCODE_BINARY)
                audio = f.read(16000)
        # 最后发送一个end of stream控制信令，通知Websocket Server音频流传输已经EOF，而后Server会主动断开连接
        endOfStreamEvent = { "endOfStream": True }
        self.send(json.dumps(endOfStreamEvent))
        global t1
        t1 = datetime.now()
        print(t1)

    _thread.start_new_thread(run_test, ())  # 启动线程执行run()函数发送数据


def test_simple():
    r"""原始的代码，没有类封装，直接跑起test_simple就能测"""
    try:
        websocket.enableTrace(False)  # True 默认在控制台打印连接和信息发送接收情况
        #ws = websocket.WebSocketApp("wss://127.0.0.1:8443/transcribe",

        ws = websocket.WebSocketApp("wss://125.217.235.84:8443/transcribe",
            on_open=on_open_test,  # 连接后自动调用发送函数
            on_message=on_message_test,  # 接收消息调用
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # 开启长连接，每run_forever一次就是执行了从on_open到on_close的整一套流程

    except Exception as e: # ws 断开 或者psycopg2.OperationalError
        traceback.print_exc()


class ASRVoiceAI:
    def __init__(self, rate=16000):
        websocket.enableTrace(False)  # True 默认在控制台打印连接和信息发送接收情况
        self.ws = websocket.WebSocketApp("wss://125.217.235.84:8443/transcribe",
            on_open=on_open,  # 连接后自动调用发送函数
            on_message=on_message,  # 接收消息调用
            on_error=on_error,
            on_close=on_close
        )
        # 这种写法就很奇怪，什么东西都往自己的成员类里面塞，而不是塞到自己身上，但是没办法，调用on_系列函数的主体还是self.ws
        self.ws.rate = rate
        self.reset()

    def trans(self, wav_data):
        r"""对一整句话进行翻译，翻译前事先知道只有一句话，最后一定是句子结束符
        支持多线程，单线程模式下取返回值作为结果，多线程模式下取ws.asr_result作为结果
        """
        self.reset()
        # 根据翻译函数选择不同的run函数
        self.ws.run = run

        self.ws.wav_data = wav_data
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # 开启长连接，每run_forever一次就是执行了从on_open到on_close的整一套流程

        return self.ws.asr_result

    def trans_stream(self, buffer, finish_record_event):
        r"""Author: zhang.haojian
        流式翻译，适合翻译包含了多个句子导致无法在翻译前完成断句的长输入
        在流式翻译过程中，recorder会不断往buffer添加音频数据，ASR模块负责读取翻译，根据Event判断音频录制是否结束
        支持多线程，单线程模式下取返回值作为结果，多线程模式下取ws.asr_result作为结果
        Args:
            buffer (list): recorder类里面的buffer，一个元素就是一个chunk
            finish_record_event (threading.Event): 结束录制的事件，由Recorder控制
        """
        self.reset()
        # 根据翻译函数选择不同的run函数
        self.ws.run = run_stream

        self.ws.buffer = buffer
        self.ws.finish_record_event = finish_record_event
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # 开启长连接，每run_forever一次就是执行了从on_open到on_close的整一套流程

        return self.ws.asr_result

    def reset(self):
        self.ws.wav_data = b""  # 准备进行ASR的音频数据
        self.ws.asr_result = ""  # ASR结果
        self.ws.buffer = []  # 用于流式ASR的缓冲区


def test():
    with open("../welcome.wav", "rb") as f:  # welcome.wav是voiceai录的采样率8000的音频
        audio = f.read()
    asr = ASRVoiceAI(8000)
    print(asr.trans(audio))


if __name__ == "__main__":
    # 跟Recorder配合使用的例子放在unit_test.py里头了
    test_simple()
