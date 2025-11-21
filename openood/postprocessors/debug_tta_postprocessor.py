from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
import scipy
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


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
        # logits = net(data)
        # preds = logits.argmax(1)
        # softmax = F.softmax(logits, 1).cpu().numpy()
        # scores = -pairwise_distances_argmin_min(
        #     softmax, np.array(self.mean_softmax_val), metric=self.kl)[1]
        # return preds, torch.from_numpy(scores)

    def set_hyperparam(self, hyperparam: list):
        print('set hyperparam', *hyperparam)
        self.temperature, self.bogus = hyperparam

    def get_hyperparam(self):
        print('get hyperparam')
        return [self.temperature, self.bogus]
