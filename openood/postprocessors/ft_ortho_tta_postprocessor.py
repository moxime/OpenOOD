from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, BatchSampler
from .ft_dist_tta_postprocessor import DistTTAPostprocessor


class OrthoTTAPostprocessor(DistTTAPostprocessor):

    def adaptation_loss(self, logits, features, net):

        bias = net.get_fc_layer().bias

        wz = logits - bias

        return wz.abs().mean(-1)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0, pred=None):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        bias = net.get_fc_layer().bias
        logits, features = net(data, return_feature=True)
        softmax_probs = torch.softmax(logits, dim=1)

        if pred is None:
            _, pred = torch.max(logits, dim=1)

        # weight.z = logits - bias
        # 1/C * sum_y |zT.m_y| = (weight.z).abs().mean(-1)
        ortho_1 = (logits - bias).abs().mean(-1)

        prob_y = torch.gather(softmax_probs, -1, pred.unsqueeze(-1)).squeeze(-1)

        # p1, p2, p3 = prob_y.log().cpu().quantile(torch.tensor([0.25, 0.5, 0.75]))
        # pm = prob_y.log().mean()
        # om = ortho_1.mean()
        # print(f'*** prob_y {p1:.3f} -- {p2:.3f} -- {p3:.3f} ({pm:.3f} ortho {om}')

        conf = prob_y.log() + ortho_1

        return pred, conf
