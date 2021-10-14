import struct
import pickle
import random

import numpy as np
import cv2

from utils.utils import EN_ZH_MAPPING


def img_preprocess(img0):
    img, ratio, (dw, dh) = letterbox(img0, 640, 32)
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # keep HWC, BGR to RGB
    return img0, img


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def get_3DPosition(depth, rgb_intrinsics, resolution=1):
    """deprecated"""
    # cx: Principal point in image, x.
    # cy: Principal point in image, y.
    # fx: Focal length x.
    # fy: Focal length y.
    # k1: k1 radial distortion coefficient
    # k2: k2 radial distortion coefficient
    # k3: k3 radial distortion coefficient
    # k4: k4 radial distortion coefficient
    # k5: k5 radial distortion coefficient
    # k6: k6 radial distortion coefficient
    # codx: Center of distortion in Z=1 plane, x (only used for Rational6KT)
    # cody: Center of distortion in Z=1 plane, y (only used for Rational6KT)
    # p2: Tangential distortion coefficient 2.
    # p1: Tangential distortion coefficient 1.
    # fmetric_radius: Metric radius.
    RESOLUTION = {{0, 0}, {1280, 720}, {1920, 1080}, {2560, 1440},
                  {2048, 1536}, {3840, 2160}, {4096, 3072}}
    cx = rgb_intrinsics[0] * RESOLUTION[resolution][0]
    cy = rgb_intrinsics[1] * RESOLUTION[resolution][1]
    fx = rgb_intrinsics[2] * RESOLUTION[resolution][0]
    fy = rgb_intrinsics[3] * RESOLUTION[resolution][1]
    xmap = np.expand_dims(np.arange(0, 480, 1), axis=1).repeat(640, axis=1)
    ymap = np.expand_dims(np.arange(0, 640, 1), axis=0).repeat(480, axis=0)
    dpt = depth
    cam_scale = 1
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    dpt = dpt.astype(np.float32) / cam_scale
    msk = (dpt > 1e-8).astype(np.float32)
    row = (ymap - cx) * dpt / fx
    col = (xmap - cy) * dpt / fy
    dpt_3d = np.concatenate(
        (row[..., None], col[..., None], dpt[..., None]), axis=2
    )
    dpt_3d = dpt_3d * msk[:, :, None]
    return dpt_3d


# def recv_data_and_load(s):
#     r"""从套接字s中收数据并转化回python的数据格式"""
#     data = s.recv(4096)
#     data = pickle.loads(data, fix_imports=True, encoding="bytes")

#     # data即为恢复出的python数据类型的数据
#     return data


# def send_data(s, data):
#     r"""把python数据类型的data通过套接字s发送出去"""
#     data = pickle.dumps(data, 0)
#     s.send(data)


def recv_data_and_load(s):
    r"""从套接字s中收数据并转化回python的数据格式，client按需请求，所以只会收到一帧数据"""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    recv_patch_len = 8192

    # 拿到本帧数据长度msg_size
    data = s.recv(payload_size)
    msg_size = struct.unpack(">L", data)[0]
    print("msg_size: {}".format(msg_size))

    while len(data) < msg_size:
        rec = s.recv(recv_patch_len)
        data += rec
    # pickle.loads的数据不应包含4字节的报头
    data = pickle.loads(data[payload_size:], fix_imports=True, encoding="bytes")

    # data即为恢复出的python数据类型的数据
    return data


def send_data(s, data):
    r"""把python数据类型的data通过套接字s发送出去"""
    data = pickle.dumps(data, 0)
    size = len(data)
    s.sendall(struct.pack(">L", size) + data)


def bgr_to_rgb(color_dict):
    r"""cv2画框的时候使用，PIL的颜色参数是RGB，cv2是BGR
    """
    for key in color_dict:
        color_dict[key] = color_dict[key][::-1]


def get_color_dict():
    r"""Author: zhang.haojian
    获取每种物体对应的颜色，用于可视化结果时给每个物体画框框
    """
    random.seed(0)
    bbox_color_dict = {
        "bottle": (57, 204, 204),
        "box": (221, 221, 221),
        "cup": (240, 18, 190),
        "kettle": (0, 116, 217),
        "pottedplant": (57, 204, 204),
    }
    for key in EN_ZH_MAPPING:
        if key in bbox_color_dict:
            continue
        color = tuple([round((random.random() * 0.6 + 0.4) * 255 + 0.5) for _ in range(3)])
        bbox_color_dict[key] = color

    return bbox_color_dict
