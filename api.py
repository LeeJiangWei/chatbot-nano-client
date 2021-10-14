import time
import re
import json
import socket
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
import threading
import urllib3
urllib3.disable_warnings()

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


def str_to_wav_bin(input_str: str) -> bytes:
    base_url = "http://125.217.235.84:18100/tts?audiotype=6&rate=1&speed=5.8&update=1&access_token=default&domain=1" \
               "&language=zh&voice_name=Jingjingcc&&text= "
    # voiceai的TTS模块目前只支持8000采样率，在播放音频时需要注意保持采样率一致
    # 当回答包括多个句子时（常见于闲聊模式），文本太长会导致对面返回空数据，所以我们要自己把数据分段发送
    result = b""
    for sentence in re.split("；|？|。|,|！|!", input_str):
        if sentence:
            r = requests.get(base_url + sentence)
            # r = requests.post(TTS_URL, json={"text": input_str})
            result += r.content
    return result


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
            "threshold": "10.0",
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
