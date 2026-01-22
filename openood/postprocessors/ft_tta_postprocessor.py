from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, BatchSampler
from .tta_postprocessor import TTAPostprocessor

from openood.losses import uniform_ce
import logging


class FTTTAPostprocessor(TTAPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature
        self.lr = self.args.lr
        self.wd = self.args.wd
        self.beta = self.args.beta
        self.auxset_size = self.args.get('auxset_size', self.chunk_size)
        self.aux_threshold = self.args.get('aux_threshold', 0)

        print(f'*** params lr={self.lr} beta={self.beta} aux thr={self.aux_threshold}')

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return
        super().setup(net, id_loader_dict, ood_loader_dict)

        self.id_train_loader = DataLoader(id_loader_dict['train'].dataset,
                                          batch_size=self.chunk_size, shuffle=True)

        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss = nn.CrossEntropyLoss(reduction='none')

        self.setup_flag = True

        self.aux_set = []

    def alternate_loss(self, logits, features, labels, net):

        return uniform_ce(logits)

    def loss_weights(self, x, logit, feature, label, where):
        """ return loss_weight, alternate_loss_weight for sample x """

        if where == 'id':
            return (1., 0.)

        if where in ('aux', 'mix'):
            return (0., self.beta)

    def reset(self, net, data_loader):
        """reset is done at each new "experiment" (dataset)

        """
        super().reset(net, data_loader)

    def new_chunk(self, net, data):

        # in super().new_chunk, model is reset
        super().new_chunk(net, data)

        # reinit id_train_tier
        self.id_train_iter = iter(self.id_train_loader)

        # calculate predicted labels on chunk
        with torch.no_grad():
            output = net(data)
            score = torch.softmax(output, dim=1)

        _, pred = torch.max(score, dim=1)
        self.chunk_predicted_labels = pred

    def update_aux_set(self, data, conf, pred, update=True):

        # if epoch not in (epochs // 2, epochs):
        #     return None

        added = []
        for x, s, y in zip(data, conf, pred):

            if s < self.aux_threshold:
                added.append(dict(data=x, pred=y, conf=s, where='aux'))

        if update:
            for _ in added:
                self.aux_set.append(_)
                if len(self.aux_set) > self.auxset_size:
                    self.aux_set.pop(0)

        return added

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        pred = self.chunk_predicted_labels

        output = net(data)
        score = torch.softmax(output, dim=1)

        conf = torch.gather(score, -1, pred.unsqueeze(-1)).squeeze(-1)

        return pred, conf

    def finetune(self, net, data, conf, pred, epoch=0, epochs=0):
        """finetune is done after postprocess (that way you can
        retrieve results before any finetuning ; that's why
        postprocess is done epochs+1 times, to benefit from n=epochs
        finetuning)

        """

        try:
            id_batch = next(self.id_train_iter)

        except StopIteration:
            self.id_train_iter = iter(self.id_train_loader)
            id_batch = next(self.id_train_iter)

        id_batch = {_: id_batch[_].cuda() for _ in ('label', 'data')}
        id_batch['pred'] = id_batch.pop('label')
        id_batch['where'] = ['id' for _ in id_batch['pred']]
        id_batch['conf'] = torch.tensor([np.inf for _ in id_batch['pred']]).cuda()

        id_list = [dict(zip(id_batch, t)) for t in zip(*id_batch.values())]
        if epoch < epochs // 2:
            batch = {'conf': conf, 'pred': pred, 'data': data, 'where': ['mix' for _ in pred]}
            mix_list = [dict(zip(batch, t)) for t in zip(*batch.values())]
        else:
            mix_list = self.update_aux_set(data, conf, pred, update=False)

        # for instance you can create a minibatch_loader

        minibatch_loader = DataLoader([*mix_list, *id_list, *self.aux_set], shuffle=True,
                                      batch_size=self.batch_size, drop_last=False)

        for batch in minibatch_loader:

            data = batch['data'].cuda()
            pred = batch['pred'].cuda()
            conf = batch['conf'].cuda()
            where = batch['where']

            count = {w: len([_ for _ in where if _ == w]) for w in set(where)}
            # print('***', count)

            logits, features = net(data, return_feature=True)
            original_loss = self.loss(logits, pred)

            alternate_loss = self.alternate_loss(logits=logits, features=features,
                                                 labels=pred, net=net)

            # loss_weights of size 2 * batch_size
            loss_weights = torch.tensor([self.loss_weights(*p)
                                         for p in zip(data, logits, features, pred, where)]).T.cuda()

            # weights_where = [*zip(where, loss_weights[0].cpu().numpy(),
            #                       loss_weights[1].cpu().numpy())]

            # print(' '.join(map(lambda t: '{}: {:.1f} / {:.1f}'.format(*t), weights_where[:10])))
            # means = ''
            # for w in ('aux', 'mix', 'id'):

            #     if w in where:
            #         i = [_ == w for _ in where]

            #         means += '{:4}: '.format(w)
            #         means += 'loss: {:.2f}+{:.2f} conf {:.2f}: --  '.format(original_loss[i].mean(),
            #                                                                 alternate_loss[i].mean())

            # print(means)

            self.optimizer.zero_grad()

            loss = (torch.vstack([original_loss, alternate_loss]) * loss_weights).sum()

            loss.backward()
            self.optimizer.step()
