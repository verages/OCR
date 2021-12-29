# -*- coding: utf-8 -*-
'''
Author: wayne
LastEditors: wayne
email: linzhihui@szarobots.com
Date: 2021-12-24 14:43:26
LastEditTime: 2021-12-27 15:55:05
Description: base ocr common and model
'''
import torch

import torch.nn as nn
from rec import build_neck
from rec import build_head
from rec import build_backbone


class BaseOCR:
    def __init__(self, config, **kwargs):
        self.config = config
        self.build_net(**kwargs)
        self.net.eval()


    def build_net(self, **kwargs):
        self.net = BaseModel(self.config, **kwargs)

    def read_pytorch_weights(self, weights_path):
        weights = torch.load(weights_path)
        return weights

    def get_out_channels(self, weights):
        out_channels = list(weights.values())[-1].numpy().shape[0]
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        # print('weighs is loaded.')


    # def inference(self, inputs):
    #     with torch.no_grad():
    #         infer = self.net(inputs)
    #     return infer

class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()

        in_channels = config.get('in_channels', 3)
        # build backbone, backbone is need for del, rec and cls
        config["Backbone"]['in_channels'] = in_channels
        self.backbone = build_backbone(config["Backbone"])
        # self.backbone  = eval("MobileNetV1Enhance")(**config["Backbone"])
        in_channels = self.backbone.out_channels

        # build neck
        self.use_neck = True
        config['Neck']['in_channels'] = in_channels
        self.neck = build_neck(config['Neck'])
        # self.neck = eval("CTCHead")(**config['Neck'], **kwargs)
        in_channels = self.neck.out_channels

        # # build head, head is need for det, rec and cls
        config["Head"]['in_channels'] = in_channels
        
        self.head = build_head(config["Head"], **kwargs)
        # self.head = eval("SequenceEncoder")(**config["Head"],**kwargs)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.backbone(x)
        if self.use_neck:
            x = self.neck(x)
        x = self.head(x)
        return x