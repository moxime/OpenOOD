from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, BatchSampler
from .ft_tta_postprocessor import FTTTAPostprocessor


class OrthoTTAPostprocessor(FTTTAPostprocessor):
    def __init__(self, config):
        super().__init__(config)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return
        super().setup(net, id_loader_dict, ood_loader_dict)

    def reset(self, net, data_loader):
        """reset is done at each new "experiment" (dataset)

        """
        return

    def new_chunk(self, net, data):
        super().new_chunk(net, data)
        self.id_train_iter = iter(self.id_train_loader)

    def alternate_loss(self, logits, features, labels, net):

        bias = net.fc.bias

        wz = logits - bias

        return wz.abs().mean(-1)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        bias = net.fc.bias
        logits, features = net(data, return_feature=True)
        softmax_probs = torch.softmax(logits, dim=1)

        pred = self.chunk_predicted_labels

        # weight.z = logits - bias
        # 1/C * sum_y |zT.m_y| = (weight.z).abs().mean(-1)

        ortho_1 = (logits - bias).abs().mean(-1)

        prob_y = torch.gather(softmax_probs, -1, pred.unsqueeze(-1)).squeeze(-1)

        # p1, p2, p3 = prob_y.log().cpu().quantile(torch.tensor([0.25, 0.5, 0.75]))
        # pm = prob_y.log().mean()
        # om = ortho_1.mean()
        # print(f'*** prob_y {p1:.3f} -- {p2:.3f} -- {p3:.3f} ({pm:.3f} ortho {om}')

        conf = prob_y.log() + ortho_1

        return self.chunk_predicted_labels, conf
