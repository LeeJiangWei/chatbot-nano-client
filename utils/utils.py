import api
import json
import time

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

TEST_INFO = [{"category": "kettle", "color": "black", "on": "diningtable", "near": "coffee machine", "material": ""},\
{"category": "cup", "color": "yellow", "on": "kitchen counter", "near": "projector", "material": "塑料或金属"},\
{"category": "tap", "color": "white", "on": "kitchen counter", "near": "", "material": "塑料或金属"}]


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

            if query_category == category:
                matched_num += 1
                if on:
                    matched_objects["on"][on] = matched_objects["on"][on] + 1 if on in matched_objects[
                        "on"].keys() else 1
                elif near:
                    matched_objects["near"][near] = matched_objects["near"][near] + 1 if near in matched_objects[
                        "near"].keys() else 1

        if matched_num != 0:
            object_description = ""

            for o in matched_objects["on"].keys():
                object_description += f"，在{o}上的有{matched_objects['on'][o]}个"

            for n in matched_objects["near"].keys():
                object_description += f"，在{n}上的有{matched_objects['near'][n]}个"

            return f"有{matched_num}个{query_category}{object_description}。"
        elif query_color:
            return f"抱歉，我没有看到{query_color}的{query_category}。"
        else:
            return f"抱歉，我没有看到{query_category}。"

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
        else:
            color_descriptions = ""
            for c in colors_of_matched.keys():
                color_descriptions += f"，{c}的有{colors_of_matched[c]}个"

            ret = f"有{matched_num + unknown_num}个{query_category}{color_descriptions}。"
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
            category, func = [EN_ZH_MAPPING[obj[i]] if obj[i] else None for i in ("category", "functions")]
            if category == query_category:
                return f"{query_category}可以用来{func}。"

        return f"抱歉，我没有看到{query_category}，所以我不知道它有什么用。"

    elif intent == "list_whats_on":
        on_objects = []
        for obj in objects:
            category, on = [EN_ZH_MAPPING[obj[i]] if obj[i] else None for i in ("category", "on")]
            if on == query_category:
                on_objects.append(category)

        if len(on_objects) > 0:
            sentence = "".join([o + "，" for o in on_objects])
            sentence = f"{query_category}上面的东西有，" + sentence[:-1] + "。"
            return sentence
        else:
            return f"对不起，我不知道{query_category}上面有什么东西。"

    elif intent == "list_whats_near":
        near_objects = []
        for obj in objects:
            category, near = [EN_ZH_MAPPING[obj[i]] if obj[i] else None for i in ("category", "near")]
            if near == query_category:
                near_objects.append(category)

        if len(near_objects) > 0:
            sentence = "".join([o + "，" for o in near_objects])
            sentence = f"{query_category}旁边的东西有，" + sentence[:-1] + "。"
            return sentence
        else:
            return f"对不起，我不知道{query_category}旁边有什么东西。"

    else:
        return "对不起，我暂时无法回答这个问题。"


def get_response(wav_data: bytes, visual_info) -> [str, [str], [bytes]]:
    response_list, wav_data_list = [], []

    t0=time.time()
    recognized_str = api.wav_bin_to_str(wav_data)
    # recognized_str = api.wav_bin_to_str_voiceai(wav_data)
    if len(recognized_str) == 0 or "没事了" in recognized_str:
        return recognized_str, None, None
    t1 = time.time()
    print("recognition:", t1 - t0)
    rasa_responses = api.question_to_answer(recognized_str)
    t2 = time.time()
    print("rasa:", t2 - t1)
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
    t3 = time.time()
    print("tts", t3 - t2)
    return recognized_str, response_list, wav_data_list


if __name__ == '__main__':
    test_query = {
        "intent": "ask_object_position",
        "object": "杯子",
        "color": ""
    }
    s = visual_to_sentence(test_query, TEST_INFO)
    print(s)
