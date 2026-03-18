from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, BatchSampler
from .ft_tta_postprocessor import FTTTAPostprocessor


class DistTTAPostprocessor(FTTTAPostprocessor):

    def __init__(self, config):

        super().__init__(config)
        assert not self.args or self.args.mu_alt in ('zero', 'mean', None)
        self.mu_alt = self.args.mu_alt if self.args else None
        print('*** mu_alt', self.mu_alt)

    def setup(self, net, *a, **kw):

        super().setup(net, *a, **kw)
        if self.mu_alt:
            if self.mu_alt == 'zero':
                self.mu_alt = torch.zeros_like(net.get_fc_layer().weight[0])
            else:
                self.mu_alt = net.get_fc_layer().weight.detach().mean(0)

    def alternate_loss(self, logits, features, net):

        return features.square().sum(-1)

    def loss_weights(self, x, logit, feature, label, conf, where, epoch=0, epochs=0):
        """ return loss_weight, alternate_loss_weight for sample x

        if mix :

        - during first half (epoch < epoch / 2): keep every sample

        - during second half: only if likely ood (conf < sb) """

        if where == 'id':
            return (self.n_samples['total'] / self.n_samples['original'], 0)

        alpha_alt = self.n_samples['total'] / self.n_samples['alt']
        if where in ('ood', 'self'):
            return (0., alpha_alt * self.beta)

        if where == 'mix':
            if epoch < epochs//2 or conf < self.pad_thresholds['self']:
                return (0., alpha_alt * self.beta)
            return (0., 0.)

        raise ValueError('{} is not a known placeholder for sample'.format(where))

    def calculate_conf(self, epoch=0, epochs=0):

        return epoch in (0, epochs//2, epochs)

    def init_epoch(self, net, data, conf, pred, epoch=0, epochs=0):

        if epoch in (0, epochs//2):
            self.reload_network(net)

        super().init_epoch(net, data, conf, pred, epoch=epoch, epochs=epochs)
        n_mix = len(conf) if epoch < epochs//2 else int((conf < self.pad_thresholds['self']).sum())

        self.n_samples = {_: len(self.pad_buffers[_]) for _ in self.pad_buffers}
        self.n_samples['mix'] = n_mix
        self.n_samples['total'] = sum(self.n_samples.values())
        self.n_samples['original'] = self.n_samples.get('id', 0)
        self.n_samples['alt'] = self.n_samples['total'] - self.n_samples['original']

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0, pred=None):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        bias = net.get_fc_layer().bias

        # weight of shape CxK (eg C=10, L=64)
        weight = net.get_fc_layer().weight

        logits, features = net(data, return_feature=True)
        softmax_probs = torch.softmax(logits, dim=1)

        if pred is None:
            _, pred = torch.max(logits, dim=1)

        b_y = bias.index_select(0, pred)  # of shape M
        m_y_2 = weight.index_select(0, pred).square().sum(-1)  # of shape M (MxL summed on last dim)

        prob_y = torch.gather(softmax_probs, -1, pred.unsqueeze(-1)).squeeze(-1)

        conf = prob_y.log() + 0.5 * (features - self.mu_alt).square().sum(dim=-1) - b_y - 0.5 * m_y_2

        return pred, conf
