import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import os
import json
import csv


APP_ID = '4a2b422c5f744f7dbf3d46db56d0f18c'
APP_SECRET = '0d0003a3a6cb4ccaaa7582c5273b2298'
url = 'https://test.finvoice.voiceaitech.com/vprc/'


def get_fileid(file_path, text='12345678'):
    '''

    :param file_path:
    :param text:
    :return:
    '''

    api = 'api/file/upload'
    file_name = os.path.basename(file_path)
    headers = {
        'x-app-id': APP_ID,
        'x-app-secret': APP_SECRET,
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
            "name": file_name,
            "text": text
        }]
    }
    # files={'file0':open(file_path,'rb')}
    multipart_encoder = MultipartEncoder(
        fields={
            'content': json.dumps(content),
            'file0': (file_name, open(file_path, 'rb'), 'audio/wav')
        },
        boundary='123456'
    )

    response = requests.post(url=url + api, data=multipart_encoder, headers=headers, verify=False).json()

    return response['data']['info_list'][0]['id']


def register_vpr(file_id, tag, group):
    """
    file_id: string or list of strings
    tag: string
    group: string
    """
    api = 'api/vpr/enroll'

    headers = {
        'x-app-id': APP_ID,
        'x-app-secret': APP_SECRET,
        'Content-Type': 'application/json'
    }
    if not isinstance(file_id, list):
        file_id = [file_id]

    content = {
        "app_id": APP_ID,
        "app_secret": APP_SECRET,
        "tag": tag,
        "group": group,  # "group.xxx"
        "model_type": "model_tird",
        "sample_rate": 16000,
        "data_check": False,
        "asr_check": False,
        "asr_model": "susie-number-16k",
        "asv_check": False,
        "asv_threshold": "0.7",
        "file_id_list": file_id
    }

    response = requests.post(url=url + api, data=json.dumps(content), headers=headers, verify=False).json()
    if response['flag'] and not response['error']:
        return True, response['data']['tag_id'], response['data']['feature_id']
    else:
        print(response['error'])
        raise Exception


def create_group(group):
    """
    group: string   ex:group.test
    """
    api = 'api/group/create'

    headers = {
        'x-app-id': APP_ID,
        'x-app-secret': APP_SECRET,
        'Content-Type': 'application/json'
    }
    content = {
        "app_id": '4a2b422c5f744f7dbf3d46db56d0f18c',
        "app_secret": '0d0003a3a6cb4ccaaa7582c5273b2298',
        "group": group
    }
    print(content)
    response = requests.post(url=url + api, data=json.dumps(content), headers=headers, verify=False).json()

    if response['flag']:
        return response['data']['group_id']
    else:
        print(response['error'])
        raise Exception


def exists_group(group):
    """
    group: string   ex:group.test
    """

    api = 'api/group/exists'

    headers = {
        'x-app-id': APP_ID,
        'x-app-secret': APP_SECRET,
        'Content-Type': 'application/json'
    }
    content = {
        "app_id": "xxx",
        "app_secret": "xxx",
        "group": group
    }
    response = requests.post(url=url + api, data=json.dumps(content), headers=headers, verify=False).json()

    if response['flag']:
        return True
    # elif response['error']['errorid']==GROUP_NOT_EXISTS:
    else:
        return False


def verify_vpr(file_id, tag, group):
    """
    file_id: string or list of strings
    tag: string
    group: string
    """
    api = 'api/vpr/identify'

    headers = {
        'x-app-id': APP_ID,
        'x-app-secret': APP_SECRET,
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

    response = requests.post(url=url + api, data=json.dumps(content), headers=headers, verify=False).json()
    if response['flag'] and not response['error']:
        print(response)
        top_tag, top_score = response['data'][0].values()
        print(top_tag,top_score)
    else:
        print(response['error'])
        raise Exception


def delete_vpr(tags, group):
    '''

    :param tags:   string or list of strings
    :param group:  string
    :return:
    '''
    api = 'api/vpr/delete'

    headers = {
        'x-app-id': APP_ID,
        'x-app-secret': APP_SECRET,
        'Content-Type': 'application/json'
    }
    if not isinstance(tags, list):
        tags = [tags]

    content = {
        "app_id": "xxx",
        "app_secret": "xxx",
        "group": group,
        "tags": tags
    }

    response = requests.post(url=url + api, data=json.dumps(content), headers=headers, verify=False).json()
    if response['flag'] and not response['error']:
        print(response)
    else:
        print(response['error'])
        raise Exception


if __name__ == '__main__':
    GROUP = 'group.test_dxx'
    if not exists_group(GROUP):
        create_group(GROUP)

    # #
    # wav_list=[line.split()[1] for line in open("wav.txt").readlines()]
    # tag2wav=[line.split() for line in open("tags.txt").readlines()]
    #
    # ### register
    # #
    # for i,t2w in enumerate(tag2wav):
    #     file_id=[]
    #     for j in range(1,len(t2w)):
    #         file_id.append(get_fileid(os.path.join('dxx2',wav_list[int(t2w[j])]+'.wav')))
    #
    #     register_vpr(file_id, t2w[0], GROUP)


    test_file='dxx2/cyx_10.wav'
    test_file_id=get_fileid(test_file)
    verify_vpr(test_file_id,'',GROUP)