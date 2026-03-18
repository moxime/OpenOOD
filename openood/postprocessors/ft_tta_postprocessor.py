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

    def reset(self, net):
        pass

    def next_aux_minibatch(self, where):

        assert where in self.stratified, '{} is ot stratified'.format(where)
        try:
            return next(self._aux_iters[where])
        except AttributeError:
            self._aux_iters = {}
        except (KeyError, StopIteration):
            self._aux_iters[where] = iter(self.aux_dls[where])

        return self.next_aux_minibatch(where)

    def alternate_loss(self, logits, features, net):

        return uniform_ce(logits)

    def loss_weights(self, x, logit, feature, label, conf, where, epoch=0, epochs=0):
        """ return loss_weight, alternate_loss_weight for sample x

        where is either id, ood, self, mix

        """

        if where == 'id':
            return (1., 0.)

        return (0., self.beta)

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
            if hasattr(self, 'n_samples'):
                print('[chunk samples]', ' -- '.join('{}:{}'.format(*t) for t in self.n_samples.items()))

    def finetune(self, net, data, conf, pred, epoch=0, epochs=0):
        """finetune is done  _epochs_ times

        """

        mix_batch = {'conf': conf, 'pred': pred, 'data': data, 'where': ['mix' for _ in pred]}
        batch_list = deque([dict(zip(mix_batch, t)) for t in zip(*mix_batch.values())])

        # for instance you can create a minibatch_loader

        minibatch_loader = DataLoader(sum((self.pad_buffers[_] for _ in self.pad_buffers), start=batch_list),
                                      shuffle=True,
                                      batch_size=self.batch_size,
                                      drop_last=False)

        if epoch == 0:
            # to track clipped_grad (removed, see below)
            self._clipped_grad = 0
            self._grad = 0

        for i, batch in enumerate(minibatch_loader):

            inspection_dict = {}

            data = batch['data'].cuda()
            pred = batch['pred'].cuda()
            conf = batch['conf'].cuda()
            where = batch['where']

            inspection_dict.update(where=where)

            if any(conf.isnan()):
                raise ValueError('{} NaN in conf'.format(conf.isnan().int().sum()))

            logits, features = net(data, return_feature=True)
            if self.n_samples['original']:
                original_loss = self.loss(logits, pred)
            else:
                original_loss = torch.zeros_like(conf)

            alternate_loss = self.alternate_loss(logits=logits, features=features, net=net)

            inspection_dict.update(original_loss=original_loss, alternate_loss=alternate_loss)
            # loss_weights of size 2 * batch_size
            p_ = zip(data, logits, features, pred, conf, where)
            weights = torch.tensor([self.loss_weights(*p, epoch, epochs)
                                   for p in p_]).T.cuda()

            inspection_dict.update(weights=weights)

            self.optimizer.zero_grad()

            loss = (original_loss * weights[0]).mean()
            loss += (alternate_loss * weights[1]).mean()

            for _ in self.stratified:
                batch_ = self.next_aux_minibatch(_)
                data = batch_['data'].cuda()
                logits, features = net(data, return_feature=True)
                if _ == 'ood':
                    w = self.beta
                    stratified_loss = self.alternate_loss(logits, features, net)
                elif _ == 'id':
                    w = 1
                    pred = batch_['label'].cuda()
                    stratified_loss = self.loss(logits, pred)

                inspection_dict.update({'stratified_loss_{}'.format(_): stratified_loss})
                loss += (w * stratified_loss).mean()

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 200)
            # if grad_norm > 200:
            #     self._clipped_grad += 1
            # self._grad += 1
            self.optimizer.step()

            flush = (i == len(minibatch_loader) - 1)
            self.inspect_minibatch(**inspection_dict, epoch=epoch, epochs=epochs, flush=flush)
