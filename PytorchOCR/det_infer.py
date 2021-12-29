# -*- coding: utf-8 -*-
'''
Author: wayne
LastEditors: wayne
email: linzhihui@szarobots.com
Date: 2021-12-28 09:30:47
LastEditTime: 2021-12-29 14:57:05
Description: 
'''
# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 10:57
# @Author  : zhoujun
import os
import sys
import pathlib
import numpy as np

# 将 torchocr路径加到python陆经里
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import torch
from torch import nn
from torchvision import transforms
from .torchocr import build_model
from .torchocr import ResizeFixedSize
from .torchocr import build_post_process


class DetInfer:
    def __init__(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        cfg = ckpt['cfg']
        self.model = build_model(cfg['model'])
        state_dict = {}
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        self.model.load_state_dict(state_dict)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.resize = ResizeFixedSize(736, False)
        self.post_process = build_post_process(cfg['post_process'])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg['dataset']['train']['dataset']['mean'], std=cfg['dataset']['train']['dataset']['std'])
        ])

    def predict(self, img, is_output_polygon=False):
        # 预处理根据训练来
        data = {'img': img, 'shape': [img.shape[:2]], 'text_polys': []}
        data = self.resize(data)
        tensor = self.transform(data['img'])
        tensor = tensor.unsqueeze(dim=0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        out = out.cpu().numpy()
        box_list, score_list = self.post_process(out, data['shape'])
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            idx = [x.sum() > 0 for x in box_list]
            box_list = [box_list[i] for i, v in enumerate(idx) if v]
            score_list = [score_list[i] for i, v in enumerate(idx) if v]
        else:
            box_list, score_list = [], []
        return box_list, score_list

def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    import cv2
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()

    pts = result[0].reshape((-1, 1, 2))
    # for point in result:
    #     point = point.astype(int)
    cv2.polylines(img_path, [pts], True, color, thickness)
    return img_path
def common_resize(src_img_cv2, _resize_h_w=(32, 32), fill_value=0):
    h, w, c = src_img_cv2.shape
    assert c == 3
    numpy_resized = np.zeros((_resize_h_w[0], _resize_h_w[1], 3), np.uint8) + fill_value
    need_scale_h = _resize_h_w[0] / float(h)
    need_scale_w = _resize_h_w[1] / float(w)
    if (need_scale_h < need_scale_w):
        resize_h = _resize_h_w[0]
        resize_w = int(w / h * _resize_h_w[0])
        numpy_data = cv2.resize(src_img_cv2, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        half_offset_w = (_resize_h_w[1] - resize_w) // 2
        numpy_resized[:, half_offset_w:half_offset_w + resize_w, :] = numpy_data
    else:
        resize_h = int(h / w * _resize_h_w[1])
        resize_w = _resize_h_w[1]
        half_offset_h = (_resize_h_w[0] - resize_h) // 2
        numpy_data = cv2.resize(src_img_cv2, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        numpy_resized[half_offset_h:half_offset_h + resize_h, :, :] = numpy_data
    return numpy_resized

if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    img = cv2.imread("./roi_image_door.jpg")
    img = common_resize(img,(224,224))
    cv2.imwrite("./1.jpg",img)
    model = DetInfer("./ch_det_mobile_db_mbv3.pth")
    box_list, score_list = model.predict(img, is_output_polygon=False)
    print(box_list)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = draw_bbox(img, box_list)
    # plt.imshow(img)
    # plt.show()
