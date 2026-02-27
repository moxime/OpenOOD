from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, BatchSampler
from .tta_postprocessor import TTAPostprocessor

from openood.losses import uniform_ce
import logging
import time


class FTTTAPostprocessor(TTAPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.batch_size = self.config.ood_dataset.batch_size

        self.setup_flag = False
        self.ft_args = self.config.postprocessor.ft
        self.lr = self.ft_args.lr
        self.wd = self.ft_args.wd
        self.beta = self.ft_args.beta

        print(f"*** params lr={self.lr} beta={self.beta} self thr={self.pad_thresholds['self']}")

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return
        super().setup(net, id_loader_dict, ood_loader_dict)

        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss = nn.CrossEntropyLoss(reduction='none')

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

    def alternate_loss(self, logits, features, labels, net):

        return uniform_ce(logits)

    def loss_weights(self, x, logit, feature, label, conf, where, epoch=0, epochs=0):
        """ return loss_weight, alternate_loss_weight for sample x """

        if where == 'id':
            return (1., 0.)

        return (0., self.beta)

    def inspect_minibatch(self, where=None, w_loss=None, epoch=0, epochs=0, **kw):
        """
        implement this methd in child class for debug purpose
        """
        if not hasattr(self, '_debug'):
            return

        if where is None or not self.calculate_conf(epoch, epochs):
            return

        wheres, wherecount = np.unique(where, return_counts=True)
        print(' -- '.join('{}: {}'.format(*_) for _ in zip(wheres, wherecount)))

        where = np.array(where)
        w_loss_w = {}
        for w in wheres:
            w_loss_w[w] = w_loss[:, where == w].mean(-1)
            print('{}: {:.2f}, {:.2f}'.format(w, *w_loss_w[w].squeeze()))

    def finetune(self, net, data, conf, pred, epoch=0, epochs=0):
        """finetune is done  _epochs_ times

        """

        mix_batch = {'conf': conf, 'pred': pred, 'data': data, 'where': ['mix' for _ in pred]}
        batch_list = [dict(zip(mix_batch, t)) for t in zip(*mix_batch.values())]

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

            data = batch['data'].cuda()
            pred = batch['pred'].cuda()
            conf = batch['conf'].cuda()
            where = batch['where']

            if any(conf.isnan()):
                raise ValueError('{} NaN in conf'.format(conf.isnan().int().sum()))

            logits, features = net(data, return_feature=True)

            original_loss = self.loss(logits, pred)

            alternate_loss = self.alternate_loss(logits=logits, features=features,
                                                 labels=pred, net=net)

            # loss_weights of size 2 * batch_size
            p_ = zip(data, logits, features, pred, conf, where)
            w_loss = torch.tensor([self.loss_weights(*p, epoch, epochs)
                                   for p in p_]).T.cuda()

            self.inspect_minibatch(where=where, w_loss=w_loss, epoch=epoch, epochs=epochs)

            self.optimizer.zero_grad()

            loss = (torch.vstack([original_loss, alternate_loss]) * w_loss).mean()

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 200)
            # if grad_norm > 200:
            #     self._clipped_grad += 1
            # self._grad += 1
            self.optimizer.step()
