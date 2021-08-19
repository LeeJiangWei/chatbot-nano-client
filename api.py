import io
import zipfile

import requests

SERVER_HOST = "gentlecomet.com"
SERVER_PORT = 8000


def get_server_response(wav_data: bytes) -> [[str], [bytes]]:
    wav_data_list = []

    r = requests.post("http://{}:{}/nano".format(SERVER_HOST, SERVER_PORT), files={"wav_data": wav_data})

    zip_container = io.BytesIO(r.content)
    zf = zipfile.ZipFile(zip_container, 'r')
    response_list = zf.namelist()

    for name in response_list:
        wav_data_list.append(zf.read(name))

    zf.close()

    return response_list, wav_data_list
