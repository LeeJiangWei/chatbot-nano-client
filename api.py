import io
import json
import socket
import zipfile

import requests

SERVER_HOST = "gentlecomet.com"
SERVER_PORT = 8000
ASR_PORT = 5050
TTS_PORT = 5051
RASA_PORT = 5005

RASA_URL = "http://{}:{}/webhooks/rest/webhook".format(SERVER_HOST, RASA_PORT)
TTS_URL = "http://{}:{}/binary".format(SERVER_HOST, TTS_PORT)
NANO_URL = "http://{}:{}/nano".format(SERVER_HOST, SERVER_PORT)


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


def get_server_response(wav_data: bytes) -> [[str], [bytes]]:
    wav_data_list = []

    r = requests.post(NANO_URL, files={"wav_data": wav_data})

    zip_container = io.BytesIO(r.content)
    zf = zipfile.ZipFile(zip_container, 'r')
    response_list = zf.namelist()

    for name in response_list:
        wav_data_list.append(zf.read(name))

    zf.close()

    return response_list, wav_data_list


def wav_bin_to_str(wav_data: bytes) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, ASR_PORT))

    buffer = ""

    fsm = asr_finite_state_machine()

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


def get_rasa_response(message: str, sender: str = "nano"):
    responses = requests.post(RASA_URL, data=json.dumps({"sender": sender, "message": message})).json()
    return responses


def str_to_wav_bin(input_str: str) -> bytes:
    r = requests.post(TTS_URL, json={"text": input_str})
    return r.content
