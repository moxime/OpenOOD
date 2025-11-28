from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, BatchSampler
from .tta_postprocessor import TTAPostprocessor


class FTTTAPostprocessor(TTAPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature
        self.lr = self.args.lr
        self.wd = self.args.wd
        self.alpha = self.args.alpha

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return
        super().setup(net, id_loader_dict, ood_loader_dict)

        self.id_train_loader = DataLoader(id_loader_dict['train'].dataset,
                                          batch_size=self.batch_size, shuffle=True)

        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss = nn.CrossEntropyLoss()
        self.alternate_loss = self._uniform_ce

        self.setup_flag = True

    @staticmethod
    def _uniform_ce(output, label, **kw):
        """ label is useless here, bu has to be in the signature of self.alternate_loss
        """
        return -output.softmax(-1).mean()

    def reset(self, net, data_loader):
        """reset is done at each new "experiment" (dataset)

        for instance, if psotprocessor manages an history dict, this
        might be the place to do it (by overriding this method)

        """
        return

    def new_chunk(self, net, data):
        super().new_chunk(net, data)
        self.id_train_iter = iter(self.id_train_loader)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        # print('*** postprocess on epoch', epoch)
        output = net(data)
        score = torch.softmax(output / self.temperature, dim=1)

        if epoch == 0:
            conf, pred = torch.max(score, dim=1)
            self.predicted_labels = pred
        else:
            conf = torch.gather(score, -1, self.predicted_labels.unsqueeze(0)).squeeze(0)

        return self.predicted_labels, conf

    def finetune(self, net, data, epoch=0):
        """finetune is done after postprocess (that way you can
        retrieve results before any finetuning ; that's why
        postprocess is done epochs+1 times, to benefit from n=epochs
        finetuning)

        """
        alpha = self.alpha

        # print('*** finetune on epoch', epoch)
        # for instance you can create a minibatch_loader
        minibatch_loader = DataLoader(list(zip(data, self.predicted_labels)), shuffle=True,
                                      batch_size=self.batch_size, drop_last=False)

        for unknown_batch in minibatch_loader:

            try:
                id_batch = next(self.id_train_iter)

            except StopIteration:
                self.id_train_iter = iter(self.id_train_loader)
                id_batch = next(self.id_train_iter)

            id_data = id_batch['data'].cuda()
            id_label = id_batch['label'].cuda()
            unknown_data, unknown_label = unknown_batch

            id_output = net(id_data)
            unknown_output = net(unknown_data)

            self.optimizer.zero_grad()

            loss = self.loss(id_output, id_label)
            loss += alpha*self.alternate_loss(unknown_output, unknown_label)

            loss.backward()
            self.optimizer.step()
