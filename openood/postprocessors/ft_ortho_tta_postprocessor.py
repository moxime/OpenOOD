from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, BatchSampler
from .ft_tta_postprocessor import FTTTAPostprocessor


class OrthoTTAPostprocessor(FTTTAPostprocessor):

    def alternate_loss(self, logits, features, labels, net):

        bias = net.get_fc_layer().bias

        wz = logits - bias

        return wz.abs().mean(-1)

    def loss_weights(self, x, logit, feature, label, conf, where, epoch=0, epochs=0):
        """ return loss_weight, alternate_loss_weight for sample x """

        if where != 'mix':
            return super().loss_weights(x, logit, feature, label, conf, where, epoch=epoch, epochs=epochs)

        """ if mix :

        - during first half (epoch < epoch / 2): keep every sample

        - during second half: only if likely ood (conf < sb) """

        return (0., self.beta if (epoch < epochs//2 or conf < self.pad_thresholds['self']) else 0.)

    def calculate_conf(self, epoch=0, epochs=0):

        return epoch in (0, epochs//2, epochs)

    def init_epoch(self, net, data, conf, pred, epoch=0, epochs=0):

        if epoch in (0, epochs//2):
            self.reload_network(net)

        if 'id' in self.pad_dls:
            batch = self.next_pad_batch('id')
            data = batch['data'].cuda()
            pred = batch['label'].cuda()
            conf = torch.inf * torch.ones_like(pred)
            self.update_pad_buffers(data, conf, pred, where='id')

        if 'ood' in self.pad_dls and not epoch:

            batch = self.next_pad_batch('ood')

            data = batch['data'].cuda()
            pred, conf = self.postprocess(net, data)

            self.update_pad_buffers(data, conf, pred, where='ood')

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
