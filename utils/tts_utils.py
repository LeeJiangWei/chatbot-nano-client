# -*- coding: utf-8 -*-
import os
import sys
import json
import websocket
import ssl
import time
from base64 import b64encode, b64decode

try:
    # 去掉当前目录，添加上级目录，如果成功说明是在进行单元测试，需要调整sys.path
    sys.path.remove(os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
except:
    # 如果失败说明是在上层进行导包，不需要调整sys.path
    pass

from utils.utils import save_wav


def on_message(self, message):
    r"""
    Args:
        message (bytes): utf-8 data received from the server
    
    response具体内容：
    参数名称        类型    描述
    code           int     错误码 4xxxx 表示客户端参数错误，5xxxx 表示服务端内部错误
    message        string  错误描述
    trace_id       string  任务 id
    data           object  合成音频片段 (带+号的都是data的键值)
      +idx         int     数据块序列号，请求内容会以流式的数据块方式返回给客户端。服务器端生成，从 1 递增
      +audio_data  string  合成的音频数据，已使用 base64 加密，客户端需进行base64 解密。
      +audio_type  string  音频类型，如 audio/pcm，audio/mp3, audio/wav
      +interval    string  音频 interval 信息，
      +end_flag    int     是否是最后一个数据块，0：否，1：是
    """
    response = json.loads(message, encoding="utf-8")
    assert self.next_idx == response["data"]["idx"], "unknown error of response['data']['idx']"
    assert response["data"]["audio_type"] == "audio/pcm", "audio type of received data is not pcm"

    self.wav_data += b64decode(response["data"]["audio_data"])

    if response["data"]["end_flag"]:  # 手里的是最后一个数据块了，后面没数据了，关掉
        self.finish_tts = True
        self.close()  # 客户端主动断开连接
    self.next_idx += 1


def on_error(self, error):
    print("------------------------------- ERROR ", error)


def on_close(self, status_code, close_msg):
    r"""
    这个函数好像只有对方关闭连接或者本地收到Ctrl+C什么的时候才会触发，自己主动关闭不会触发，不是很懂
    Args:
        status_code (int): 状态码，如1000, 1011
        close_msg (str): 返回的信息，貌似正常关闭时都是个空字符串
    """
    pass
    # print("------------------------------- CLOSE ")


def on_open(self):
    data = {  # 参数挺多，目前这些就够用了，其他的详见文档
        "access_token": "default",
        "version": "1.0",
        "tts_params": {
            "domain": "1",
            "interval": "0",
            "language": "zh",
            "voice_name": "Jingjingcc",
            "audiotype": "5",
            "speed": "5.6",
            # 文本要加密，加密必须是bytes，但是json字符串只能有str，所以加密完又要decode一下
            "text": b64encode(self.text.encode(encoding="utf-8")).decode(),
        },
    }
    self.send(json.dumps(data))  # 建立连接之后先发送配置，然后才能开始发语音


class TTSBiaobei:
    r"""Author: zhang.haojian
    标贝科技的TTS接口，流式输出，不需要再手动去分句整流水线
    """
    def __init__(self):
        websocket.enableTrace(False)  # True 默认在控制台打印连接和信息发送接收情况
        self.ws = websocket.WebSocketApp("ws://125.217.235.84:19008",
            on_open=on_open,  # 连接后自动调用发送函数
            on_message=on_message,  # 接收消息调用
            on_error=on_error,
            on_close=on_close
        )
        self.reset()

    def start_tts(self, text):
        # 把整个websocket对象都给返回出去了，后面拿着它去访问wav_data
        self.reset()
        self.ws.text = text
        # run_forever()并没有新开线程，它是阻塞式地执行一系列on函数，直到连接关闭，所以我们要在外面自己开多线程
        self.ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # 开启长连接，每run_forever一次就是执行了从on_open到on_close的整一套流程
        return self.ws

    def reset(self):
        # 这种写法就很奇怪，什么东西都往自己的成员类里面塞，而不是塞到自己身上，但是没办法，调用on_系列函数的主体还是self.ws
        self.ws.text = ""  # 准备进行转换的文本
        self.ws.wav_data = b""  # 存放TTS结果
        self.ws.next_idx = 1  # 用于标记目前收到哪个块了，网络不好的时候也许会用上
        self.ws.finish_tts = False  # 标记TTS是否已经结束，外部音频播放器会用到


if __name__ == "__main__":
    text = "妈妈给六岁的儿子出算术题做。“你一共有六个苹果，爸爸拿走两个，妈妈拿走四个，你还剩几个苹果？”儿子听后很激动说：“我真的是亲生的吗？”"
    tts = TTSBiaobei()
    ws = tts.start_tts(text)
    while not ws.finish_tts:
        time.sleep(0.1)
    save_wav(ws.wav_data, "tmp.wav", channels=1, rate=8000)
