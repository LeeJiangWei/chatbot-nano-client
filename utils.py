import api
import json

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
    "dining table": "桌子",
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
    "green": "绿色",
    "red": "红色",
    "purple": "紫色",
    "black": "黑色",
    "yellow": "黄色",
    "blue": "蓝色",
    "white": "白色"
}

SYNONYM = {
    "杯子": ("水杯", "茶杯", "咖啡杯"),
    "水壶": ("茶壶", "烧水壶"),
    "桌子": ("餐桌",)
}

TEST_INFO = '[{"category": "kettle", "color": "black", "on": "dining table", "near": "coffee machine"},\
{"category": "cup", "color": "yellow", "on": "kitchen counter", "near": "projector"},\
{"category": "tap", "color": "white", "on": "kitchen counter", "near": ""}]'


def visual_to_sentence(query, info):
    intent, query_category, query_color, query_on, query_near = [query[i] for i in
                                                                 ("intent", "object", "color", "on", "near")]

    objects = json.loads(info)
    for obj in objects:
        category, color, on, near = [EN_ZH_MAPPING[obj[i]] if obj[i] else None
                                     for i in ("category", "color", "on", "near")]

        if (category == query_category or category in SYNONYM.keys() and query_category in SYNONYM[category]) \
                and (not query_color or color == query_color):
            sentence = category
            if intent == "ask_object_position":
                if on:
                    sentence += f"在{on}上，"
                if near:
                    sentence += f"在{near}旁边，"
                if query_color:
                    sentence = f"{color}的" + sentence
                elif color:
                    sentence = sentence + f"它是{color}的，"
            elif intent == "ask_object_color":
                sentence += f"是{color}的。"

            return sentence

    return f"没有看到{query_color if query_color else ''}{query_category}。"


def get_response(wav_data: bytes, visual_info) -> [str, [str], [bytes]]:
    response_list, wav_data_list = [], []

    recognized_str = api.wav_bin_to_str(wav_data)

    rasa_responses = api.question_to_answer(recognized_str)
    for response in rasa_responses:
        if "text" in response.keys():
            text = response['text']
        elif "custom" in response.keys():
            text = visual_to_sentence(response['custom'], visual_info)
        else:
            text = ""

        wav = api.str_to_wav_bin(text)
        response_list.append(text)
        wav_data_list.append(wav)

    return recognized_str, response_list, wav_data_list
