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
class DebugTTAPostprocessor(BasePostprocessor):
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

        print('ID LOADER DICT')
        _unfold_(id_loader_dict)

        print('OOD LOADER DICT')
        _unfold_(ood_loader_dict)

        self.setup_flag = True

    def process_batch(self, net, data):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data) / (self.temperature + self.bogus)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):

        pred_list, conf_list, label_list = super().inference(net, data_loader, progress)

        batch_size = data_loader.batch_size

        df = pd.DataFrame(columns=range(min(label_list), max(label_list) + 1))

        for i in range(len(data_loader)):

            batch_label = label_list[i * batch_size:(i + 1) * batch_size]

            counts = np.unique(batch_label, return_counts=True)

            df.loc[len(df)] = {s: c for s, c in zip(*counts)}

        df = df.fillna(0).astype(int)

        ind_cols = [_ for _ in df if _ >= 0]
        ood_cols = [_ for _ in df if _ < 0]

        df['ind'] = df[ind_cols].sum(axis=1)
        df['ood'] = df[ood_cols].sum(axis=1)

        cols = ['ind', *ood_cols]
        print(df[cols].to_string())

        return pred_list, conf_list, label_list

    def set_hyperparam(self, hyperparam: list):
        print('set hyperparam', *hyperparam)
        self.temperature, self.bogus = hyperparam

    def get_hyperparam(self):
        print('get hyperparam')
        return [self.temperature, self.bogus]
