from .ft_tta_postprocessor import FTTTAPostprocessor
from torch.utils.data import DataLoader, BatchSampler
import torch.nn as nn
import torch
import numpy as np
from typing import Any


class DistTTAPostprocessor(FTTTAPostprocessor):

    def __init__(self, config):

        super().__init__(config)
        assert not self.args or self.args.mu_ood in ('zero', 'mean', None)
        self.mu_ood = self.args.mu_ood if self.args else None
        self.iterations = self.args.iterations_per_phase
        print('*** mu_ood', self.mu_ood)

    def setup(self, net, *a, **kw):

        super().setup(net, *a, **kw)
        if self.mu_ood:
            if self.mu_ood == 'zero':
                self.mu_ood = torch.zeros_like(net.get_fc_layer().weight[0])
            else:
                self.mu_ood = net.get_fc_layer().weight.detach().mean(0)

    def adaptation_loss(self, logits, features, net):

        return features.square().sum(-1)

    def new_chunk(self, net, data, epochs=0):

        super().new_chunk(net, data, epochs=epochs)

        if not self.iterations:
            self.switch_phase = epochs//4
            return

        padded_mix_size = len(data) + sum(len(self.pad_buffers[_]) for _ in self.pad_buffers)

        it_per_epoch = padded_mix_size / self.batch_size

        self.switch_phase = int(np.ceil(self.iterations / it_per_epoch))

        self.max_iterations_on['padded_mix'] = 2 * self.iterations

        _s = 'epochs: {} it/epoch: {} it: {}'
        assert epochs * it_per_epoch >= 2 * self.iterations, _s.format(epochs, it_per_epoch, 2 * self.iterations)

    def losses_weight(self, conf=0., where='id', epoch=0, epochs=0, **kw):
        """ return id_loss_weight, adaptation_loss_weight for sample x

        if mix :

        - during first half (epoch < epoch / 4): keep every sample

        - during second half: only if likely ood (conf < sb) """

        if where == 'mix':
            if epoch < self.switch_phase or conf < self.pad_thresholds['self']:
                return (0., self.beta)
            return (0., 0.)

        return super().losses_weight(conf=conf, where=where, epoch=epoch, epochs=epochs, **kw)

        raise ValueError('{} is not a known placeholder for sample'.format(where))

    def calculate_conf(self, epoch=0, epochs=0):

        return epoch in (0, self.switch_phase, epochs)

    def init_epoch(self, net, data, conf, pred, epoch=0, epochs=0):

        if epoch in (0, self.switch_phase):
            self.reload_network(net)

        super().init_epoch(net, data, conf, pred, epoch=epoch, epochs=epochs)

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

        conf = prob_y.log() + 0.5 * (features - self.mu_ood).square().sum(dim=-1) - b_y - 0.5 * m_y_2

        return pred, conf
