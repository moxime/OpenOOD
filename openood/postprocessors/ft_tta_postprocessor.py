from typing import Any

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, BatchSampler
from .tta_postprocessor import TTAPostprocessor

from .batch_inspector import BatchInspector

from openood.losses import uniform_ce, MarginCrossEntropy
import logging
import time


class FTTTAPostprocessor(TTAPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.batch_size = self.config.ood_dataset.batch_size

        self.stratified = [_ for _ in ('id', 'ood') if self.config.postprocessor.padding.get(_, 0) < 0]

        self.setup_flag = False
        self.ft_args = self.config.postprocessor.ft
        self.lr = self.ft_args.lr
        self.wd = self.ft_args.wd
        self.beta = self.ft_args.beta

        self.filterout_null_weights = self.ft_args.only_non_zeros
        self.size_normalization = True

        self.max_iterations_on = {}

        print(f"*** params lr={self.lr} beta={self.beta} self thr={self.pad_thresholds['self']}")

    def setup(self, net: nn.Module, id_loader_dict, id_ood_loader_dict):
        if self.setup_flag:
            return
        super().setup(net, id_loader_dict, id_ood_loader_dict)

        for _ in self.stratified:
            self.aux_dls[_] = id_ood_loader_dict['aux'][_]

        self.loss = MarginCrossEntropy(margin=self.config.network.margin, reduction='none')
        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.setup_flag = True

        unfreeze = self.config.postprocessor.ft.unfreeze
        if unfreeze:
            for _ in net.parameters():
                _.requires_grad_(False)

            for _ in net.get_fc_layer().parameters():
                _.requires_grad_(True)

        if unfreeze in ('all', 'penultimate'):
            for name, submodule in reversed(list(net.named_children())):
                for p in submodule.parameters():
                    p.requires_grad_(True)

                if name.lower().startswith('layer') and unfreeze == 'penultimate':
                    break

    def next_aux_minibatch(self, where):

        assert where in self.stratified, '{} is ot stratified'.format(where)
        try:
            return next(self._aux_iters[where])
        except AttributeError:
            self._aux_iters = {}
        except (KeyError, StopIteration):
            self._aux_iters[where] = iter(self.aux_dls[where])

        return self.next_aux_minibatch(where)

    def adaptation_loss(self, logits, features, net):

        return uniform_ce(logits)

    def losses_weight(self, data, conf, label, where, epoch=0, epochs=0, **kw):
        """ return loss_weight, adaptation_loss_weight for sample data

        where is either id, ood, self, mix

        """

        if where == 'id':
            return (1., 0.)

        return (0., self.beta)

    def loss_weights(self, data, conf, pred, where, epoch=0, epochs=0, size_normalization=False):
        """ returns  a len(data)x2 tensor of original and loss weights

        where can be in (id, ood, mix, self)
        """

        if not isinstance(where, (list, tuple)):
            where = [where for _ in conf]

        raw_weights = torch.tensor([self.losses_weight(*p, epoch, epochs)
                                    for p in zip(data, conf, pred, where)]).cuda()
        if not size_normalization:
            return raw_weights

        normalization = torch.ones(2, device=conf.device)

        n_id_orig_pos = (raw_weights[[w == 'id' for w in where], 0] > 0).sum()
        n_ood_mix_self_alt_pos = (raw_weights[[w != 'id' for w in where], 1] > 0).sum()

        normalization[0] = len(conf) / n_id_orig_pos
        normalization[1] = len(conf) / n_ood_mix_self_alt_pos

        return normalization.unsqueeze(0) * raw_weights

    def inspect_minibatch(self, epoch=0, epochs=0, flush=False, **kw):
        """
        implement this methd in child class for debug purpose
        """
        if not hasattr(self, '_debug'):
            return

        if not hasattr(self, '_inspector'):
            self._inspector = BatchInspector()

        self._inspector.epoch = epoch
        self._inspector.iterations += 1
        if self.calculate_conf(epoch, epochs):
            self._inspector.update_mb(epoch, epochs=epochs, flush=flush, **kw)

        if flush and epoch == epochs - 1:
            print('[chunk] {} it / {} epochs = {} it /epoch'.format(self._inspector.iterations,
                                                                    epochs,
                                                                    self._inspector.iterations / epochs))
            print('[chunk samples]', ' -- '.join('{}:{}'.format(*t) for t in self.samples_dist.items()))

    def finetune(self, net, data, conf, pred, epoch=0, epochs=0):
        """finetune is done  _epochs_ times

        """
        for _, m in self.max_iterations_on.items():
            if self.iterations_on[_] >= m:
                return

        mix_batch = {'conf': conf, 'pred': pred, 'data': data, 'where': ['mix' for _ in pred]}
        batch_list = [dict(zip(mix_batch, t)) for t in zip(*mix_batch.values())]

        for where in self.pad_buffers:
            batch_list.extend(self.pad_buffers[where])

        for d in batch_list:
            d['id_weight'] = self.losses_weight(**d, epoch=epoch, epochs=epochs)[0]
            d['adaptation_weight'] = self.losses_weight(**d, epoch=epoch, epochs=epochs)[1]

        if self.filterout_null_weights:
            batch_list = [_ for _ in batch_list if batch_list['id_weight'] or batch_list['adaptation_weight']]

        minibatch_loader = DataLoader(batch_list,
                                      shuffle=True,
                                      batch_size=self.batch_size,
                                      drop_last=False)

        if epoch == 0:
            # to track clipped_grad (removed, see below)
            self._clipped_grad = 0
            self._grad = 0

        for i, batch in enumerate(minibatch_loader):

            self.iterations_on['padded_mix'] = self.iterations_on.get('padded_mix') + 1
            inspection_dict = {}

            data = batch['data'].cuda()
            pred = batch['pred'].cuda()
            conf = batch['conf'].cuda()
            where = batch['where']

            inspection_dict.update(where=where)

            if any(conf.isnan()):
                raise ValueError('{} NaN in conf'.format(conf.isnan().int().sum()))

            logits, features = net(data, return_feature=True)
            id_loss = self.loss(logits, pred)

            adaptation_loss = self.adaptation_loss(logits=logits, features=features, net=net)

            inspection_dict.update(original_loss=id_loss, adaptation_loss=adaptation_loss)
            # loss_weights of size 2 * batch_size
            batch_weights = self.loss_weights(data, conf, pred, where,
                                              epoch=epoch, epochs=epochs,
                                              size_normalization=True).T

            inspection_dict.update(weights=batch_weights)

            self.optimizer.zero_grad()

            loss = (id_loss * batch_weights[0]).mean()
            loss += (adaptation_loss * batch_weights[1]).mean()

            for _ in self.stratified:
                self.iterations_on['stratified_'+_] = self.iterations_on.get('stratified_'+_) + 1
                batch_ = self.next_aux_minibatch(_)
                data = batch_['data'].cuda()
                logits, features = net(data, return_feature=True)
                if _ == 'ood':
                    w = self.beta
                    stratified_loss = self.adaptation_loss(logits, features, net)
                elif _ == 'id':
                    w = 1
                    pred = batch_['label'].cuda()
                    stratified_loss = self.loss(logits, pred)

                inspection_dict.update({'stratified_loss_{}'.format(_): stratified_loss})
                loss += (w * stratified_loss).mean()

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 200)
            if grad_norm > 200:
                self._clipped_grad += 1
            self._grad += 1
            self.optimizer.step()

            flush = (i == len(minibatch_loader) - 1)
            self.inspect_minibatch(**inspection_dict, epoch=epoch, epochs=epochs, flush=flush)
