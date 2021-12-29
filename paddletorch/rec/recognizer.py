# -*- coding: utf-8 -*-
'''
Author: wayne
LastEditors: wayne
email: linzhihui@szarobots.com
Date: 2021-12-24 14:43:26
LastEditTime: 2021-12-27 15:55:06
Description: TextRecognizer
'''
import cv2
import numpy as np
import math
import torch
from rec.baseocr import BaseOCR
from rec.postprocess import CTCLabelDecode


class TextRecognizer(BaseOCR):
    def __init__(self, rec_model_path, **kwargs):
        self.weights_path = rec_model_path
        rec_image_shape = [3,32,320]
        self.rec_image_shape = [int(v) for v in rec_image_shape]
        self.rec_batch_num = 6
        rec_char_dict_path = ('./rec/ppocr_keys_v1.txt')
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_type": "ch",
            "character_dict_path": rec_char_dict_path,
            "use_space_char": True
        }
        self.postprocess_op = eval("CTCLabelDecode")(**postprocess_params)

        self.use_gpu = torch.cuda.is_available()

        self.limited_max_width = 1280
        self.limited_min_width = 16

        network_config = {'Backbone': {'name': 'MobileNetV1Enhance', 'scale': 0.5},
                        'Neck': {'name': 'SequenceEncoder', 'hidden_size': 64, 'encoder_type': 'rnn'},
                        'Head': {'name': 'CTCHead', 'mid_channels': 96, 'fc_decay': 2e-05}}
        weights = self.read_pytorch_weights(self.weights_path)
        self.out_channels = self.get_out_channels(weights)
        kwargs['out_channels'] = self.out_channels
        super(TextRecognizer, self).__init__(network_config, **kwargs)

        self.load_state_dict(weights)
        self.net.eval()
        if self.use_gpu:
            self.net.cuda()

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        max_wh_ratio = max(max_wh_ratio, imgW / imgH)
        imgW = int((32 * max_wh_ratio))
        imgW = max(min(imgW, self.limited_max_width), self.limited_min_width)
        h, w = img.shape[:2]
        ratio = w / float(h)
        ratio_imgH = math.ceil(imgH * ratio)
        ratio_imgH = max(ratio_imgH, self.limited_min_width)
        if ratio_imgH > imgW:
            resized_w = imgW
        else:
            resized_w = int(ratio_imgH)
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im




    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                # h, w = img_list[ino].shape[0:2]
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            # starttime = time.time()

            with torch.no_grad():
                inp = torch.from_numpy(norm_img_batch)
                if self.use_gpu:
                    inp = inp.cuda()
                prob_out = self.net(inp)
            preds = prob_out.cpu().numpy()

        rec_result = self.postprocess_op(preds)
        for rno in range(len(rec_result)):
            rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        # elapse += time.time() - starttime
        return rec_res
