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
from .ft_tta_postprocessor import FTTTAPostprocessor
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
class DebugTTAPostprocessor(FTTTAPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False
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

    def update_pad_set(self, data, conf, pred, where='self', **kw):

        n = super().update_pad_set(data, conf, pred, where=where, **kw)
        # print('*** added {} pad samples from {}'.format(n, where))
        return n

    def loss_weights(self, x, logit, feature, label, conf, where, epoch=0, epochs=0):
        w = super().loss_weights(x, logit, feature, label, conf, where, epoch=epoch, epochs=epochs)

        # if epoch in (epochs // 3, 2 * epochs // 3):
        #     print('{} {:4} conf={:.1f}: {:.1f}, {:.1f}'.format(epoch, where, conf, *w))
        return w
