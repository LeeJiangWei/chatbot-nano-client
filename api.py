import io
import json
import socket
import zipfile
import os
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests


SERVER_HOST = "gentlecomet.com"
ASR_PORT = 5050
TTS_PORT = 5051
RASA_PORT = 5005

RASA_URL = "http://{}:{}/webhooks/rest/webhook".format(SERVER_HOST, RASA_PORT)
TTS_URL = "http://{}:{}/binary".format(SERVER_HOST, TTS_PORT)

# voiceprint
VPR_APP_ID = '4a2b422c5f744f7dbf3d46db56d0f18c'
VPR_APP_SECRET = '0d0003a3a6cb4ccaaa7582c5273b2298'
VPR_URL = 'https://test.finvoice.voiceaitech.com/vprc/'
VPR_GROUP = 'group.test_dxx'


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
    sock.connect((SERVER_HOST, ASR_PORT))

    buffer = ""

    fsm = ASRFSM()

    sock.send(wav_data)
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


def question_to_answer(message: str, sender: str = "nano"):
    responses = requests.post(RASA_URL, data=json.dumps({"sender": sender, "message": message})).json()
    return responses


def str_to_wav_bin(input_str: str) -> bytes:
    r = requests.post(TTS_URL, json={"text": input_str})
    return r.content


class VoicePrint:
    def __init__(self):

        # TODO: load tag2name from file

        self.tag2name = {'test0': '路人甲', 'test1': '张三', 'test2': '王大力','lwh':"刘伟恒"}

    @staticmethod
    def get_fileid_bin(wav_data, text='12345678'):
        '''

        :param file_path:
        :param text:  unimportant if asr_check=False
        :return:
        '''

        API = 'api/file/upload'
        # file_name = os.path.basename(file_path)
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
                "name": 'wake.wav',
                "text": text
            }]
        }
        # files={'file0':open(file_path,'rb')}
        multipart_encoder = MultipartEncoder(
            fields={
                'content': json.dumps(content),
                'file0': ('wake.wav', wav_data, 'audio/wav')
            },
            boundary='123456'
        )

        response = requests.post(url=VPR_URL + API, data=multipart_encoder, headers=headers, verify=False).json()

        return response['data']['info_list'][0]['id']

    @staticmethod
    def verify_vpr(file_id, tag, group):
        """
        file_id: string or list of strings
        tag: string  if "",  verify 1:N
        group: string
        """
        API = 'api/vpr/identify'

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
            "threshold": "60.0",
            "ext": False,
            "sorting": True,
            "top_n": 10,
            "file_id_list": file_id
        }

        response = requests.post(url=VPR_URL + API, data=json.dumps(content), headers=headers, verify=False).json()
        if response['flag'] and not response['error']:
            if len(response['data']) != 0:
                print(response)
                top_tag, top_score = response['data'][0].values()
                print(top_tag,top_score)
                return top_tag, top_score
            else:
                print("Unregister VoicePrint")
                return "stranger", 0.0
        else:
            print(response['error'])
            raise Exception

    def get_spk_name(self, wav_data):
        file_id = self.get_fileid_bin(wav_data)
        top_tag, top_score = self.verify_vpr(file_id, tag='', group=VPR_GROUP)

        if top_tag in self.tag2name.keys():
            return self.tag2name[top_tag]
        else:
            return "客人"
