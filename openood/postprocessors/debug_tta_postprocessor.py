from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
import scipy
from tqdm import tqdm

from torch.utils.data import DataLoader


from .base_postprocessor import BasePostprocessor
from .tta_postprocessor import TTAPostprocessor
from .info import num_classes_dict
import openood.utils.comm as comm

import pandas as pd


def _unfold_(obj, prefix='', prefix_inc='      '):

    if isinstance(obj, (list, tuple)):
        for _ in obj:
            _unfold_(obj, prefix=prefix+prefix_inc, prefix_inc=prefix_inc)

        return

    if isinstance(obj, dict):

        for _ in obj:
            print(prefix + _)
            _unfold_(obj[_], prefix=prefix+prefix_inc, prefix_inc=prefix_inc)

        return

    str_ = [type(obj)]

    if isinstance(obj, torch.utils.data.dataloader.DataLoader):
        str_l = '{N}= {n}x{m} s: {s} from '.format(n=len(obj), m=obj.batch_size,
                                                   N=len(obj) * obj.batch_size,
                                                   s=type(obj.sampler).__name__[0])
        d_str_ = str(obj.dataset).split('\n')

        str_ = [(' ' * len(str_l) if i else str_l) + _ for i, _ in enumerate(d_str_)]

    for _ in str_:
        print(prefix + _)


# A TTA PostProcessor to look at batches
class DebugTTAPostprocessor(TTAPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature
        self.bogus = self.args.bogus

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return
        super().setup(net, id_loader_dict, ood_loader_dict)

        print('ID LOADER DICT')
        _unfold_(id_loader_dict)

        print('OOD LOADER DICT')
        _unfold_(ood_loader_dict)

        self.setup_flag = True
