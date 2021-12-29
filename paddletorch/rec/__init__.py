# -*- coding: utf-8 -*-
'''
Author: wayne
LastEditors: wayne
email: linzhihui@szarobots.com
Date: 2021-12-24 14:43:26
LastEditTime: 2021-12-25 14:27:01
Description: init
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.nn import modules


__all__ = ['build_head','build_neck',"build_backbone"]

from .neckhead import SequenceEncoder,CTCHead
from .backbone import MobileNetV1Enhance


def build_head(config, **kwargs):

    module_class = eval("CTCHead")(**config, **kwargs)
    return module_class

def build_neck(config):
    module_class = eval("SequenceEncoder")(**config)
    return module_class

def build_backbone(config):
    module_class = eval("MobileNetV1Enhance")(**config)
    return module_class