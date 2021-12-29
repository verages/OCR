# -*- coding: utf-8 -*-
'''
Author: wayne
LastEditors: wayne
email: linzhihui@szarobots.com
Date: 2021-12-28 09:30:47
LastEditTime: 2021-12-29 14:50:40
Description: 
'''
from .augment import *
import copy
from addict import Dict
from .DBPostProcess import DBPostProcess
from .DetModel import DetModel

support_post_process = ['DBPostProcess','DetModel']


def build_post_process(config):
    """
    get architecture model class
    """
    copy_config = copy.deepcopy(config)
    post_process_type = copy_config.pop('type')
    assert post_process_type in support_post_process, f'{post_process_type} is not developed yet!, only {support_post_process} are support now'
    post_process = eval(post_process_type)(**copy_config)
    return post_process



def build_model(config):
    """
    get architecture model class
    """
    copy_config = copy.deepcopy(config)
    arch_type = copy_config.pop('type')
    assert arch_type in support_post_process, f'{arch_type} is not developed yet!, only {support_post_process} are support now'
    arch_model = eval(arch_type)(Dict(copy_config))
    return arch_model
