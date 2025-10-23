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


class TTAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False
        print('*** INIT ***')
        print(config)
        print('*** END OF INIT ***')

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        self.setup_flag = True

    def process_batch(self, net, data):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits = net(data)
        preds = logits.argmax(1)
        softmax = F.softmax(logits, 1).cpu().numpy()
        scores = -pairwise_distances_argmin_min(
            softmax, np.array(self.mean_softmax_val), metric=self.kl)[1]
        return preds, torch.from_numpy(scores)
