# Chatbot-nano-client

## 环境配置

1. 安装 [miniforge-pypy3](https://github.com/conda-forge/miniforge) 作为 conda 环境

2. 安装 `pyaudio` 包

   ```bash
   sudo apt-get update
   sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev
   
   git clone http://people.csail.mit.edu/hubert/git/pyaudio.git
   cd ./pyaudio
   python setup.py install
   ```

3. 安装 `tensorflow` 包

   请参考：[Installing TensorFlow For Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html) ，tensorflow 版本为 1.10 以上应该都可以（不能为2），重要的是要按照开发板对应的 Jetpack 版本，见[Release Notes For Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/index.html) ，我们这边无法得知 Jetpack 版本且没办法重新刷一个，安装遇到了困难。

   > https://forums.developer.nvidia.com/t/how-to-check-the-jetpack-version/69549
   >
   > Given a TX2 board, there is no way to check which JetPack version it is flashed with. You can only check its L4T info. For example, cat /etc/nv_tegra_release on board.
   
4. 其余依赖直接用pip安装即可，matplotlib 可以不装，不装的话在 main.py 中把绘图代码删掉即可。

## 程序逻辑 


+ `main.py` 主事件循环 
    + 唤醒后开始录音
    + 判断声音变化（静音时间超过阈值）停止录音
    + 发送录音到服务端，返回输出音频，直接播放
    + 音频加入播放队列
+ `api.py` 向服务器请求 
  + 一次完成识别、对话、合成

+ `classifier.py`唤醒

## 服务器相关端口和示例

服务器地址：http://gentlecomet.com

### 语音识别（ASR）

端口：5050

示例代码

```python
import socket

class asr_finite_state_machine:
    def __init__(self):
        self.curr_state = 'start'
        # trans 0: chinese or " " , 1: "/r" , 2: "/n"
        self.state_trans = {
            'start': ['recogn', 'start', 'start'],
            'recogn': ['recogn', 'start', 'end'],
            'end': ['end', 'end', 'end']
        }

    def set_start(self):
        self.curr_state = 'start'

    def get_state(self):
        return self.curr_state

    def trans(self, word):
        # trans: 0
        if '\u4e00' <= word <= '\u9fff' or word == " ":
            self.curr_state = self.state_trans[self.curr_state][0]
        # trans: 1
        elif word == '\r':
            self.curr_state = self.state_trans[self.curr_state][1]
        # trans : 2
        elif word == '\n':
            self.curr_state = self.state_trans[self.curr_state][2]

            
def wav_bin_to_str(wav_data: bytes) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ASR_HOST, ASR_PORT))

    buffer = ""

    fsm = asr_finite_state_machine()

    sock.send(wav_data)
    received_byte = sock.recv(2048)
    received_str = str(received_byte, encoding="utf-8")
    words = list(received_str)
    while received_str != "":
        print(received_byte)
        print(received_str)
        print("-" * 80)

        for word in words:
            fsm.trans(word)
            state = fsm.get_state()

            if state == 'start':
                buffer = ''
            elif state == 'recogn':
                buffer = buffer + word
            elif state == 'end':
                buffer = buffer.replace(" ", "")
                print("Final Recognized Result: ", buffer)
                sock.close()
                return buffer
        # buffer = received_str

        received_byte = sock.recv(2048)
        received_str = str(received_byte, encoding="utf-8")
        words = list(received_str)

    sock.close()
    buffer = buffer.replace(" ", "")
    print("Final Recognized Result: ", buffer)
    return buffer
```

### 对话

请求路径：`http://gentlecomet.com:8000/webhooks/rest/webhook`

示例代码

```python
import requests
import json

def get_rasa_response(message: str, sender: str = "server"):
    """
    Send message to rasa server and get response
    :param message: String to be sent
    :param sender: String that identify sender
    :return: List of dicts
    """
    responses = requests.post(BASE_URL, data=json.dumps({"sender": sender, "message": message})).json()
    return responses
```

### 语音合成（TTS）

请求路径：`http://gentlecomet.com:5051/binary`

示例代码

```python
import requests
import io
import soundfile as sf

def str_to_wav_bin(input_str: str) -> bytes:
    r = requests.post("http://gentlecomet.com:5051/binary", json={"text": input_str})
    wav = r.content

    container = io.BytesIO(wav)

    y, sr = sf.read(container)
    sf.write("mywav.wav", y, sr)
    
    return y
```
