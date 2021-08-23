import api


def get_response(wav_data: bytes) -> [str, [str], [bytes]]:
    response_list, wav_data_list = [], []

    recognized_str = api.wav_bin_to_str(wav_data)

    rasa_responses = api.get_rasa_response(recognized_str)
    for response in rasa_responses:
        if "text" in response.keys():
            text = response['text']
            wav = api.str_to_wav_bin(text)

            response_list.append(text)
            wav_data_list.append(wav)

    return recognized_str, response_list, wav_data_list
