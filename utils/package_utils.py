# 这里存放了跟前端UI互动相关的一些函数
import threading
import time
import json
import gzip
import time
from base64 import b64encode

import keyboard

EXPIRE_TIME = 30


class Package(object):

    @staticmethod
    def real_time_detection_result(img_data):
        r"""本数据包包含视觉模块的检测结果"""
        assert type(img_data) == str
        return {
            'type': 0,
            'data': img_data
        }

    @staticmethod
    def real_time_state(state):
        r"""本数据包是对前端心跳包请求的回应，回复后端当前的状态"""
        assert type(state) == int
        return {
            'type': 1,
            'data': state
        }

    @staticmethod
    def voice_to_word_result(data, success=True):
        r"""本数据包包含用户输入的语音转成文字的结果"""
        # if success: data may be the sentence of speaking. else: data may be the error message e.g. "network is error"
        assert type(success) == bool
        return {
            'type': 2,
            'success': success,
            'data': data
        }

    @staticmethod
    def response_word_result(data, success=True):
        r"""本数据包包含智能系统回复的文本结果"""
        # if success: data may be the sentence of response. else: data may be the error message e.g. "network is error"
        assert type(success) == bool
        return {
            'type': 3,
            'success': success,
            'data': data
        }


class ReceivePackage(object):
    def __init__(self, sock, addr, main_process):
        self.sock = sock
        self.addr = addr
        self.main_process = main_process
        return

    def __call__(self, data):
        try:
            data = json.loads(data)
            assert 'type' in data
        except:
            return False
        if data['type'] == 0:
            return self.start_speech(data)
        elif data['type'] == 1:
            return self.close_speech(data)
        elif data['type'] == 2:
            return self.exit_reading(data)
        elif data['type'] == 3:
            return self.heart_beat(data, self.main_process.clients)
        else:
            return False

    def start_speech(self, *args):
        self.main_process.start_speech()
        return True

    def close_speech(self, *args):
        self.main_process.close_speech()
        return True

    def exit_reading(self, *args):
        self.main_process.close_reading()
        return True

    def heart_beat(self, data, clients):
        if 'time' not in data or type(data['time']) != float:
            return False
        if str(self.addr) in clients:
            if clients[str(self.addr)]['last_active_time'] < float(data['time']):
                clients[str(self.addr)]['last_active_time'] = float(data['time'])
        else:
            clients[str(self.addr)] = {
                'sock': self.sock,
                'addr': self.addr,
                'last_active_time': float(data['time'])
            }
        return True


def sendPackage(sock, data):
    assert type(data) == dict
    data = json.dumps(data).strip() + '\n'
    try:
        sock.send(data.encode('utf-8'))
    except:
        pass
    return


def groupSendPackage(data, clients):
    r"""将数据包群发给每一个前端，返回仍在线的客户端dict"""
    activate_clients = {}
    for addr in clients:
        if time.time() - clients[addr]['last_active_time'] > EXPIRE_TIME:
            print(addr, 'disconnected')
            clients[addr]['sock'].close()
        else:
            sendPackage(clients[addr]['sock'], data)
            activate_clients[addr] = clients[addr]
    clients = activate_clients
    return


def receivePackage(sock):
    history_data = b''
    while True:
        data = sock.recv(1)
        if data == b'':
            break
        if data == b'\n':
            yield history_data
            history_data = b''
        else:
            history_data += data
    return


def transform_for_send(bytes_data):
    r"""因为json只支持string不支持bytes，为了网络传输方便把图像原始的二进制数据转换成string"""
    return b64encode(gzip.compress(bytes_data, 6)).decode("utf-8")


def client_service(sock, addr, main_process, ):
    main_process.clients[str(addr)] = {
        'sock': sock,
        'addr': addr,
        'last_active_time': time.time()
    }
    # send initialization state to connecting socket.
    sendPackage(sock, Package.real_time_state(main_process.state))
    sendPackage(sock, Package.real_time_detection_result(main_process.detection_result))

    rp = ReceivePackage(sock, addr, main_process)
    for message in receivePackage(sock):
        rp(message)
    return


def wait_keyboard(interrupt_event):
    r"""Author: zhang.haojian
    在多线程的场景下使用，监听键盘输入，在按下某个特定按键时通过Event向外部传达信息，由外部进行相应处理后
    重置该Event
    （结果发现监听键盘不是在后端搞的，是前端监听，监听到特定按键时发信号过来，所以这个函数就用不上了，尴尬）
    Args:
        interrupt_event (thread.Event): 用于记录是否按下打断键，可以打断播音或者录音
    """
    while True:
        keyboard.wait("Esc")
        interrupt_event.set()


def test_wait_keyboard():
    r"""wait_keyboard()函数的单元测试，在无按键输入时应当持续输出running，在按下Esc时应当输出
    detected Esc!，而后继续输出running
    """
    interrupt_event = threading.Event()
    s = threading.Thread(target=wait_keyboard, args=(interrupt_event,))
    s.setDaemon(True)
    s.start()
    while True:
        print("running")
        time.sleep(0.1)
        if interrupt_event.is_set():
            print("detected Esc!")
            interrupt_event.clear()


if __name__ == "__main__":
    test_wait_keyboard()
