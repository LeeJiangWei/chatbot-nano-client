import io
import re
import time
import wave

import pyaudio

import api

SYNONYM_TABLE = {
    "鼻子": "杯子",
    "水浒": "水壶",
    "台版": "白板",
    "黑板选": "黑板刷",
    "微波楼": "微波炉",
    "一只": "椅子",
    "一直": "椅子",
    "停止": "瓶子",
}

EN_ZH_MAPPING = {
    "person": "人",
    "bicycle": "",
    "car": "",
    "motorbike": "",
    "aeroplane": "",
    "bus": "",
    "train": "",
    "truck": "",
    "boat": "",
    "traffic light": "",
    "fire hydrant": "",
    "stop sign": "",
    "parking meter": "",
    "bench": "",
    "bird": "",
    "cat": "",
    "dog": "",
    "horse": "",
    "sheep": "",
    "cow": "",
    "elephant": "",
    "bear": "",
    "zebra": "",
    "giraffe": "",
    "backpack": "",
    "umbrella": "",
    "handbag": "手提包",
    "tie": "",
    "suitcase": "",
    "frisbee": "",
    "skis": "",
    "snowboard": "",
    "sports ball": "",
    "kite": "",
    "baseball bat": "",
    "baseball glove": "",
    "skateboard": "",
    "surfboard": "",
    "tennis racket": "",
    "bottle": "瓶子",
    "wine glass": "",
    "cup": "杯子",
    "fork": "",
    "knife": "",
    "spoon": "",
    "bowl": "",
    "banana": "",
    "apple": "",
    "sandwich": "",
    "orange": "",
    "broccoli": "",
    "carrot": "",
    "hot dog": "",
    "pizza": "",
    "donut": "",
    "cake": "",
    "chair": "椅子",
    "sofa": "",
    "pottedplant": "盆栽",
    "bed": "",
    "diningtable": "桌子",
    "toilet": "",
    "tvmonitor": "",
    "laptop": "",
    "mouse": "",
    "remote": "遥控器",
    "keyboard": "",
    "cell phone": "",
    "microwave": "微波炉",
    "oven": "",
    "toaster": "",
    "sink": "",
    "refrigerator": "",
    "book": "",
    "clock": "",
    "vase": "",
    "scissors": "",
    "teddy bear": "",
    "hair drier": "",
    "toothbrush": "",
    "bin": "垃圾桶",
    "blackboard eraser": "黑板擦",
    "box": "箱子",
    "coffee beans": "咖啡豆",
    "coffee grinder": "磨豆机",
    "coffee machine": "咖啡机",
    "kettle": "水壶",
    "kitchen counter": "厨柜",
    "power strip": "排插",
    "projector": "投影仪",
    "tap": "水龙头",
    "whiteboard": "白板",
    "ground": "地",
}

COLOR_MAPPING = {
    "green": "绿色",
    "red": "红色",
    "purple": "紫色",
    "black": "黑色",
    "yellow": "黄色",
    "blue": "蓝色",
    "white": "白色"
}

TEST_INFO = [{"category": "kettle", "color": "black", "on": "diningtable", "near": "coffee machine", "material": ""},
             {"category": "cup", "color": "", "on": "kitchen counter", "near": "projector", "material": "塑料或金属"},
             {"category": "cup", "color": "white", "on": "diningtable", "near": "", "material": "塑料或金属"},
             {"category": "cup", "color": "white", "on": "kitchen counter", "near": "", "material": "塑料或金属"}]


def visual_to_sentence(query, objects):
    intent, query_category, query_color = [query[i] for i in ("intent", "object", "color",)]

    if not query_category:
        return "对不起，我没听清楚您的问题。"

    if intent == "ask_object_position":
        matched_objects = {"on": {}, "near": {}}
        matched_num = 0

        # sequentially search objects that match query
        for obj in objects:
            category, on, near = [EN_ZH_MAPPING[obj[i]] if obj[i] else None for i in ("category", "on", "near")]
            color = obj["color"]

            # query的物体出现在objects中 AND (query中未指定color OR query指定了color且与物体的color一致) 视为成功匹配
            if query_category == category and (not query_color or query_color == color):
                matched_num += 1
                if on:
                    matched_objects["on"][on] = matched_objects["on"][on] + 1 if on in matched_objects[
                        "on"].keys() else 1
                elif near:
                    matched_objects["near"][near] = matched_objects["near"][near] + 1 if near in matched_objects[
                        "near"].keys() else 1

        if matched_num == 0:
            if query_color:
                return f"抱歉，我没有看到{query_color}的{query_category}。"
            else:
                return f"抱歉，我没有看到{query_category}。"
        elif matched_num == 1:  # 匹配的物体只有一个，则不回复它的数量
            object_description = ""
            for o in matched_objects["on"].keys():
                object_description += f"在{o}上"

            for n in matched_objects["near"].keys():
                object_description += f"在{n}旁边"

            if query_color:
                return f"{query_color}的{query_category}{object_description}。"
            else:
                return f"{query_category}{object_description}。"
        else:
            object_description = ""
            on_dict = matched_objects["on"]
            near_dict = matched_objects["near"]

            if len(on_dict) + len(near_dict) == 1:  # 所有物体都出现在同一个参照物的上面或旁边，就用总结性的句子
                for o in on_dict.keys():
                    object_description += f"，它们都在{o}上面"

                for n in near_dict.keys():
                    object_description += f"，它们都在{n}旁边"
            else:  # 超过2个参照物，用排比的句子
                for o in on_dict.keys():
                    object_description += f"，在{o}上的有{matched_objects['on'][o]}个"

                for n in near_dict.keys():
                    object_description += f"，在{n}旁边的有{matched_objects['near'][n]}个"

            if query_color:
                return f"有{matched_num}个{query_color}的{query_category}{object_description}。"
            else:
                return f"有{matched_num}个{query_category}{object_description}。"

    elif intent == "ask_object_color":
        colors_of_matched = {}
        unknown_num = 0
        for obj in objects:
            category, on, near = [EN_ZH_MAPPING[obj[i]] if obj[i] else None for i in ("category", "on", "near")]
            color = obj["color"]

            if query_category == category:
                if color:
                    colors_of_matched[color] = colors_of_matched[color] + 1 if color in colors_of_matched.keys() else 1
                else:
                    unknown_num += 1

        matched_num = sum(list(colors_of_matched.values()))
        if matched_num + unknown_num == 0:
            return f"抱歉，我没有看到{query_category}。"
        elif matched_num == 0:
            return f"我看到了{matched_num + unknown_num}个{query_category}，但是我不认识它们的颜色。"
        elif matched_num == 1:
            return f"{query_category}是{list(colors_of_matched.keys())[0]}的。"
        elif len(colors_of_matched) == 1 and unknown_num == 0:
            color_description = ""
            for c in colors_of_matched.keys():
                color_description += f"它们都是{c}的"
            return f"有{matched_num}个{query_category}，{color_description}。"
        else:
            color_description = ""
            for c in colors_of_matched.keys():
                color_description += f"，{c}的有{colors_of_matched[c]}个"

            ret = f"有{matched_num + unknown_num}个{query_category}{color_description}。"
            if unknown_num > 0:
                ret += f"还有{unknown_num}个不知道是啥颜色。"

            return ret

    elif intent == "ask_object_quantity":
        counter = 0
        for obj in objects:
            category, on, near = [EN_ZH_MAPPING[obj[i]] if obj[i] else None for i in ("category", "on", "near")]
            color = obj["color"]

            if query_category == category and (not query_color or query_color == color):
                counter += 1

        if counter == 0:
            if query_color:
                return f"抱歉，我没有看到{query_color}的{query_category}。"
            else:
                return f"抱歉，我没有看到{query_category}。"
        else:
            if query_color:
                return f"{query_color}的{query_category}有{counter}个。"
            else:
                return f"{query_category}有{counter}个。"

    elif intent == "ask_object_material":
        for obj in objects:
            category, on, near = [EN_ZH_MAPPING[obj[i]] if obj[i] else None for i in ("category", "on", "near")]
            material = obj["material"]
            color = obj["color"]

            if query_category == category:
                if query_color:
                    if color and color == query_color:
                        if material:
                            return f"{query_color}的{query_category}是{material}的。"
                        else:
                            return f"抱歉，我看不出来{query_color}的{query_category}是什么材料做的噢。"
                    else:
                        break
                else:
                    if material:
                        return f"{query_category}是{material}的。"
                    else:
                        return f"抱歉，我看不出来{query_category}是什么材料做的噢。"

        if query_color:
            return f"抱歉，我没有看到{query_color}的{query_category}。"
        else:
            return f"抱歉，我没有看到{query_category}。"

    elif intent == "ask_object_function":
        for obj in objects:
            category = EN_ZH_MAPPING[obj["category"]]
            func = obj["function"]
            if category == query_category:
                return f"{query_category}可以用来{func}。"

        return f"抱歉，我没有看到{query_category}，所以我不知道它有什么用。"

    elif intent == "list_whats_on":
        on_objects = {}
        for obj in objects:
            category, on = [EN_ZH_MAPPING[obj[i]] if obj[i] else None for i in ("category", "on")]
            if on == query_category:
                on_objects[category] = on_objects[category] + 1 if on in on_objects.keys() else 1

        if len(on_objects) > 0:
            sentence = "，".join([f"{on_num}个{on}" for on, on_num in on_objects.items()])
            return f"{query_category}的上面有{sentence}。"
        else:
            return f"对不起，我不知道{query_category}上面有什么东西。"

    elif intent == "list_whats_near":
        near_objects = {}
        for obj in objects:
            category, near = [EN_ZH_MAPPING[obj[i]] if obj[i] else None for i in ("category", "near")]
            if near == query_category:
                near_objects[category] = near_objects[category] + 1 if near in near_objects.keys() else 1

        if len(near_objects) > 0:
            sentence = "，".join([f"{near_num}个{near}" for near, near_num in near_objects.items()])
            return f"{query_category}的旁边有{sentence}。"
        else:
            return f"对不起，我不知道{query_category}旁边有什么东西。"

    elif intent == "list_all":
        all_objects = {}
        for obj in objects:
            category = EN_ZH_MAPPING[obj["category"]]
            all_objects[category] = all_objects[category] + 1 if category in all_objects.keys() else 1

        return "我看到了" + "，".join([f"{n}个{o}" for o, n in all_objects.items()]) + "。"

    else:
        return "对不起，我暂时无法回答这个问题。"


def get_response(wav_data: bytes, visual_info) -> [str, [str], [bytes]]:
    response_list, wav_data_list = [], []

    t0 = time.time()
    recognized_str = api.wav_bin_to_str(wav_data)
    # recognized_str = api.wav_bin_to_str_voiceai(wav_data)

    for k, v in SYNONYM_TABLE.items():
        recognized_str = recognized_str.replace(k, v)

    if len(recognized_str) == 0 or "没事了" in recognized_str:
        return recognized_str, None, None
    t1 = time.time()
    print("recognition:", t1 - t0)
    rasa_responses = api.question_to_answer(recognized_str)
    t2 = time.time()
    print("rasa:", t2 - t1)
    for response in rasa_responses:
        if "text" in response.keys():
            text = response["text"]
        elif "custom" in response.keys():
            text = visual_to_sentence(response["custom"], visual_info)
        else:
            text = ""

        wav = api.str_to_wav_bin(text)
        response_list.append(text)
        wav_data_list.append(wav)
    t3 = time.time()
    print("tts", t3 - t2)
    return recognized_str, response_list, wav_data_list


def synonym_substitution(recognized_str):
    for k, v in SYNONYM_TABLE.items():
        recognized_str = recognized_str.replace(k, v)
    return recognized_str


def remove_punctuation(recognized_str):
    punctuations = "；|？|。|,|！|!"
    for k in punctuations:
        recognized_str = recognized_str.replace(k, "")
    return recognized_str


def get_answer(input_text, visual_info):
    r"""根据输入的文本，判断其意图并给出相应的答复
    Return:
        text (str): 回应的原始文本，用于发送给前端显示
        sentences (list): 每个元素是一个句子的文本，用于转音频和播音
    """
    # 先前的设计是rasa的responses可能包含多个元素，比如调用api前回一句，调用api后再把真正的回复说出来，现在只是一轮回复，所以直接取[0]
    response = api.question_to_answer(input_text)[0]
    text = ""
    if "text" in response.keys():
        text = response["text"]
    elif "custom" in response.keys():
        text = visual_to_sentence(response["custom"], visual_info)
    text = text.replace("~", "")  # ~符号在句子末尾时TTS模块会给出突然的静音，用户体验不好（在句子中间倒还算正常）

    # 当回答包括多个句子时（常见于闲聊模式），文本太长会导致TTS服务器返回空数据，所以我们要自己把数据分段发送
    sentence = re.split("；|？|。|,|！|!", text) if text else []

    return text, sentence


def bytes_to_wav_data(bytes_data, format=pyaudio.paInt16, channels=1, rate=16000):
    r"""Author: zhang.haojian 
    给字节流数据加上wav文件头，成为完整的wav格式数据
    Args:
        bytes_data (bytes): input bytes data
    Return:
        wav_data (bytes): data that add wav format head
    """
    # 通过wave库将字节流写入文件中再读回来，数据中就有了wav格式头(比输入多44个字节)
    # 借用io库的缓冲区，不实际写入文件读取文件，而是在缓冲区操作，速度更快
    container = io.BytesIO()
    with wave.open(container, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(bytes_data)
    container.seek(0)
    wav_data = container.read()

    return wav_data


def save_wav(wav_data, filepath, format=pyaudio.paInt16, channels=1, rate=16000):
    r"""Author: zhang.haojian
    把二进制数据保存成wav文件
    Args:
        wav_data (bytes): 音频数据流，标准情况下不含wav格式头，含了也没有影响，因为只是音频
            最前面多了44个字节的数据，人是听不出来的
        filepath (str): wav文件的保存路径
        format, channels, rate: 数据格式，声道数，采样率，略
    """
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(wav_data)


if __name__ == "__main__":
    test_query = {
        "intent": "ask_object_color",
        "object": "水壶",
        "color": ""
    }
    s = visual_to_sentence(test_query, TEST_INFO)
    print(s)
