# 跟调接口相关的函数和类都放在这里
import time
import re
import json
import socket
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import threading
import urllib3
urllib3.disable_warnings()
import random

SERVER_HOST = "222.201.134.203"
# SERVER_HOST = "gentlecomet.com"
# ASR_SERVER_HOST="222.201.137.105"
ASR_SERVER_HOST="222.201.134.203"
# ASR_SERVER_HOST = 'localhost' #kaldi, only for debuging
ASR_PORT = 15050  # kaldi. NOTE: 大学城是5050，五山由于防火墙，用的是15050
TTS_PORT = 5051
RASA_PORT = 17003

SAMPLE_RATE = 16000  # 8000|16000
PAUSE_TIME = 200  # [50, 500]

RASA_URL = "http://{}:{}/webhooks/rest/webhook".format(SERVER_HOST, RASA_PORT)
TTS_URL = "http://{}:{}/binary".format(SERVER_HOST, TTS_PORT)

# voiceprint
VPR_APP_ID = '4a2b422c5f744f7dbf3d46db56d0f18c'
VPR_APP_SECRET = '0d0003a3a6cb4ccaaa7582c5273b2298'
VPR_URL = 'https://test.finvoice.voiceaitech.com/vprc/'
VPR_GROUP = 'group.demo'


class ASRFSM:
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
    sock.connect((ASR_SERVER_HOST, ASR_PORT))

    buffer = ""

    fsm = ASRFSM()

    sock.send(wav_data)
    sock.shutdown(socket.SHUT_WR)
    received_byte = sock.recv(2048)
    received_str = str(received_byte, encoding="utf-8")
    words = list(received_str)
    while received_str != "":
        for word in words:
            fsm.trans(word)
            state = fsm.get_state()

            if state == 'start':
                buffer = ''
            elif state == 'recogn':
                buffer = buffer + word
            elif state == 'end':
                buffer = buffer.replace(" ", "")
                sock.close()
                return buffer

        received_byte = sock.recv(2048)
        received_str = str(received_byte, encoding="utf-8")
        words = list(received_str)

    sock.close()
    buffer = buffer.replace(" ", "")
    return buffer


def sender(wave_data: bytes, sock:socket.socket):
    # Simulating real time says. Of course you can send the whole paragraph at once.
    # 模拟实时聊天，不等一句话说完再发，而是连续发送，由服务端进行断句
    sock.sendall(wave_data)
    sock.shutdown(socket.SHUT_WR)
    return


def receive_as_generator(sock):
    # To prevent sockets from sending sticky packets, \n is used here as a separator. e.g. {...}\n{...}\n
    # 为了防止两个数据包粘在一起，在两个数据包中间加入一个\n
    data = b''
    while True:
        try:
            content = sock.recv(1024)
        except Exception as e:
            if str(e) == "timed out":
                # 当wav_data是无人声数据时，服务器没有识别到汉字，将不会返回任何信息，此时客户端需要自己结束这次连接
                # 在超过timeout时长（目前为20s）后，sock将raise一个超时异常，本函数给出一个结束符，让外部循环结束连接
                yield b'{"alternatives": [{"transcript": ""}], "is_final": true}'
            else:
                raise e

        if content == b'':
            break
        end_pos = content.find(b'\n')  # 一个数据包结束的位置
        while end_pos != -1:  # 有结束符
            data += content[:end_pos]
            yield data
            data = b''
            content = content[end_pos + 1:]
            end_pos = content.find(b'\n')
        # 一直切分到data没有结束符
        data += content


def wav_bin_to_str_voiceai(wav_data: bytes) -> str:
    r"""voiceai ASR模块1.0版本的调用方式"""
    sock = socket.socket()
    sock.settimeout(3)  # 3s内没有收到服务端的翻译结果就断开，作为当我们客户端发送噪声片段时的一个容错手段，防止一直阻塞在这里
    sock.connect(("125.217.235.84", 8635))
    sock.send(json.dumps({"sample_rate": SAMPLE_RATE, "pause_time": PAUSE_TIME}).encode("utf-8"))
    config_result = sock.recv(1024)  # here will return back the message that is the result of configurature.
    print(config_result[:-1])
    try:
        config_result = json.loads(config_result[:-1])
    except:
        print("data is wrong")
        exit()
    if "success" not in config_result or config_result["success"] != 1:
        print("configuration error. msg:" + str(config_result["msg"]))
        exit()

    s = threading.Thread(target=sender, args=(wav_data, sock))
    s.setDaemon(True)
    s.start()

    for msg in receive_as_generator(sock):
        msg = json.loads(msg)
        print(msg)
        # is_final表示到达句子的末尾，此时的翻译结果是一整句的翻译结果，在is_final=False时，拿到的结果是不完整的
        if msg["is_final"]:
            sock.close()
            return msg["alternatives"][0]["transcript"]

    sock.close()


def question_to_answer(message: str, sender: str = "nano"):
    responses = requests.post(RASA_URL, data=json.dumps({"sender": sender, "message": message})).json()
    return responses


def str_to_wav_bin(input_str: str, speed=5.8) -> bytes:
    # speed: 语速，[0.0, 9.0]，默认5.0
    base_url_get = f"http://125.217.235.84:18100/tts?audiotype=6&rate=1&speed={speed}&update=1&access_token=default&domain=1" \
               "&language=zh&voice_name=Jingjingcc&&text= "
    base_url_post = "http://125.217.235.84:18100/tts"
    # voiceai的TTS模块目前只支持8000采样率，在播放音频时需要注意保持采样率一致
    # NOTE: 据说语音名称改成Jingjing就可以16K了，不过我还没试
    # 当回答包括多个句子时（常见于闲聊模式），文本太长会导致对面返回空数据，所以我们要自己把数据分段发送
    method = "get"
    result = b""
    for sentence in re.split("；|？|。|,|！|!", input_str):
        if sentence:  # GET方法限制不超过250 UTF-8字符，超过就会返回空数据，太多要用POST
            if method == "get":
                r = requests.get(base_url_get + sentence)
            elif method == "post":
                from ipdb import set_trace
                set_trace()
                data = {
                    "access_token": "default",
                    "domain": "1",
                    "language": "zh",
                    "voice_name": "Jingjingcc",
                    "audiotype": "6",
                    "rate": "1",
                    "speed": f"{speed}",
                    "update": "1",
                    "text": sentence,
                }
                r = requests.post(base_url_post, data=data)
            result += r.content
    return result


def str_to_wav_bin_unblock(sentences, wav_data_queue, finish_stt_event) -> bytes:
    r"""流式（非阻塞式）地输出每个句子的翻译结果，而不是所有句子翻译完才一起输出结果
    Args:
        sentences (list): member of MainProcess class, sentences to be STT
        wav_data_queue (list): member of MainProcess class, a queue to input speech wav data after TTS
        finish_stt_event (threading.Event()): event indicates whether STT is finished
    """
    base_url = "http://125.217.235.84:18100/tts?audiotype=6&rate=1&speed=5.8&update=1&access_token=default&domain=1" \
               "&language=zh&voice_name=Jingjingcc&&text= "
    # voiceai的TTS模块目前只支持8000采样率，在播放音频时需要注意保持采样率一致
    while sentences:
        sentence = sentences.pop(0)
        # GET方法限制不超过250 UTF-8字符，超过就会返回空数据，太多要用POST
        r = requests.get(base_url + sentence)
        wav_data_queue.append(r.content)
    finish_stt_event.set()

    return


class VoicePrint:
    def __init__(self):
        self.tag2name = json.load(open("./spk_name.json"))

    @staticmethod
    def get_fileid_bin(wav_data, text='12345678'):
        '''上传音频数据，获取fileid

        :param file_path:
        :param text:  unimportant if asr_check=False
        Return:
            file_id (str): 32 bits id of file
        '''

        upload_api = 'api/file/upload'
        headers = {
            'x-app-id': VPR_APP_ID,
            'x-app-secret': VPR_APP_SECRET,
            'Content-Type': 'multipart/form-data;boundary=123456'
        }
        content = {
            "app_id": "xxx",
            "app_secret": "xxx",
            "vad_check": False,
            "asv_check": False,
            "asv_threshold": "0.52",
            "asr_check": False,
            "asr_model": "susie-number-16k",
            "action_type": "0",
            "info": [{
                "name": 'wake.wav',  # 音频数据在对端服务器上保存的文件名
                "text": text  # 对该音频文件的描述文本
            }]
        }
        # 使用multipart/form-data的方式传送正文数据时，数据需要先放进一个MultipartEncoder
        multipart_encoder = MultipartEncoder(
            fields={
                'content': json.dumps(content),
                'file0': ('wake.wav', wav_data, 'audio/wav')
            },
            boundary='123456'
        )

        t1 = time.time()
        response = requests.post(url=VPR_URL + upload_api, data=multipart_encoder, headers=headers, verify=False).json()
        t2 = time.time()
        print(f"获取fileid的时间为{t2 - t1:.2f}秒")

        return response['data']['info_list'][0]['id']

    @staticmethod
    def verify_vpr(file_id, tag, group):
        """
        file_id: string or list of strings
        tag: string  if "",  verify 1:N
        group: string
        """
        verify_api = 'api/vpr/identify'

        headers = {
            'x-app-id': VPR_APP_ID,
            'x-app-secret': VPR_APP_SECRET,
            'Content-Type': 'application/json'
        }
        if not isinstance(file_id, list):
            file_id = [file_id]

        content = {
            "app_id": "xxx",
            "app_secret": "xxx",
            "tag": tag,
            "group": group,
            "model_type": "model_tird",
            "sample_rate": 16000,
            "data_check": False,
            "asr_check": False,
            "asr_model": "susie-number-16k",
            "asv_check": False,
            "asv_threshold": "0.7",
            "threshold": "25.0",
            "ext": False,
            "sorting": True,
            "top_n": 10,
            "file_id_list": file_id
        }

        t2 = time.time()
        response = requests.post(url=VPR_URL + verify_api, data=json.dumps(content), headers=headers, verify=False).json()
        t3 = time.time()
        print(f"获取声纹识别结果的时间为{t3 - t2:.2f}秒")
        if response['flag'] and not response['error']:
            if len(response['data']) != 0:
                print("声纹识别结果：", response)
                top_tag, top_score = response['data'][0].values()
                return top_tag, top_score
            else:
                print("Unregistered VoicePrint")
                return "unknown", 0.0
        else:
            print(response['error'])
            raise Exception

    def get_spk_name(self, wav_data):
        file_id = self.get_fileid_bin(wav_data)
        top_tag, top_score = self.verify_vpr(file_id, tag="", group=VPR_GROUP)

        if top_tag in self.tag2name.keys():
            return self.tag2name[top_tag]
        else:  # 遇到不认识的都当做客人，后续可能会接入注册声纹的功能
            return self.tag2name['unknown']


ERROR_RESPONSE = [
    "你在说啥呀，我怎么听不懂捏。",
    "这就触及到我的知识盲区啦。",
    "刚才一不留神没听清楚你说了啥，能再说一次吗。",
    "哈哈哈，这是啥意思。",
]

class BaiduDialogue():
    r"""Author: li.jiangwei & zhang.haojian
    原本放在rasa模块，现在搬到client的百度聊天API
    """
    def __init__(self):
        with open("./profile.json") as f:
            profile = json.load(f)
        self.key = profile["baidu_api_key"]
        self.secret = profile["baidu_api_secret"]
        self.service_id = profile["baidu_service_id"]

        self.access_token = self.get_access_token()
        self.session_id = ""
        print("Baidu dialogue successfully initialized. Access token: ", self.access_token)

        self.chat_response = ""  # 保存聊天结果，用于多线程
        # self.last_request_time = time.time()  # 保存上一次请求的时间，确保两次请求相隔超过1秒，因为我们的服务QPS限额是1

    def get_access_token(self):
        resp = requests.get("https://aip.baidubce.com/oauth/2.0/token", params={
            "grant_type": "client_credentials",
            "client_id": self.key,
            "client_secret": self.secret
        }).json()

        return resp["access_token"]

    def run(
            self, input_text
    ):
        self.chat_response = ""
        print("Input to turing bot: ", input_text)

        body = {
            "version": "3.0",
            "service_id": self.service_id,
            "log_id": "server",
            "session_id": self.session_id,
            "request": {
                "terminal_id": "00000001",
                "query": input_text
            }
        }

        url = f"https://aip.baidubce.com/rpc/2.0/unit/service/v3/chat?access_token={self.access_token}"

        # NOTE: 不知道为啥，设到1.22秒有时候还是会报错，好像这个时间并没有估计准确，好比上一次数据在路上走了2s，
        # api处理了0.5s，花0.2s发回来，我们看到的是两次访问相差2.7s，然后第二次请求在路上走了0.2s，对服务器来说
        # 两次请求相差0.9s，因此还是会报QPS错误，所以这个时间不好估计，先不做处理，反正对话时间肯定长过1s了
        # time.sleep(max(0, 1.22 - (time.time() - self.last_request_time)))  # 保持两次请求间隔大于1秒
        # self.last_request_time = time.time()
        response = requests.post(url, json=body).json()

        error_code = response["error_code"]
        if error_code != 0:
            # see: https://ai.baidu.com/ai-doc/UNIT/qkpzeloou#%E9%94%99%E8%AF%AF%E4%BF%A1%E6%81%AF
            err_msg = response["error_msg"]

            print(f"Baidu dialogue error {error_code}: ", err_msg)
            text = random.choice(ERROR_RESPONSE)
        else:
            result = response["result"]

            self.session_id = result["session_id"]
            if "actions" not in result["responses"][0]:
                # 一般是QPS超限，常发生于测试时，用于整个系统应该不会到这里
                text = "你说话太快啦，坐下来喝杯水吧"
            else:
                text = result["responses"][0]["actions"][0]["say"]

            if text[-1] not in "，。？！“”：；":
                text += "。"
            # print("Output from baidu bot: ", text)
        self.chat_response = text
        return self.chat_response


def test_baidu_dialogue():
    data = {
        "weather": [
            "今天天气怎么样？",
            "广东广州",
            "明天天气怎么样？",
            "深圳福田区",
            "后天天气怎么样？"
        ],
        "location": [
            "广州在哪里？",
            "上海在哪里？",
            "北京在哪里？"
        ],
        "chat": [
            "讲一个笑话吧~",
            "讲个笑话吧！"
        ],
        "want": [
            "我想喝水。",
            "我想喝咖啡",
            "我想睡觉",
            "我想休息"
    ]}

    B = BaiduDialogue()
    for key, query_texts in data.items():
        print(f"测试百度API对{key}类问题的回答..")
        for query_text in query_texts:
            t1 = time.time()
            answer = B.run(query_text)
            print(f"{query_text + ' ' * (25 - len(query_text.encode('gbk')))} --> {answer}")
            time.sleep(0.5)  # 保证QPS<1
            print(f"time: {time.time() - t1:.2f}s")


def test():
    input_texts = [
        "今天天气怎么样？",  # out_of_scope
        "广东广州",  # out_of_scope
        "北京在哪里？",  # out_of_scope
        "讲一个笑话吧~",  # out_of_scope
        "我想喝咖啡",  # out_of_scope
        "桌子在哪里",  # ask_object_position
        "微波炉是什么颜色的呢？",  # ask_object_color
        "有几个箱子？",  # ask_object_quantity
        "水龙头有什么功能呀？",  # ask_object_function
        "你看到了什么东西",  # list_all
    ]

    for input_text in input_texts:
        response = question_to_answer(input_text)[0]
        print(response)


if __name__ == "__main__":
    test()
    test_baidu_dialogue()
