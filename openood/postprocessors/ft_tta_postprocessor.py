from typing import Any

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, BatchSampler
from .tta_postprocessor import TTAPostprocessor

from openood.losses import uniform_ce


class FTTTAPostprocessor(TTAPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.setup_flag = False
        self.args = self.config.postprocessor.postprocessor_args
        self.temperature = self.args.temperature
        self.lr = self.args.lr
        self.wd = self.args.wd
        self.beta = self.args.beta

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return
        super().setup(net, id_loader_dict, ood_loader_dict)

        self.id_train_loader = DataLoader(id_loader_dict['train'].dataset,
                                          batch_size=self.batch_size, shuffle=True)

        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss = nn.CrossEntropyLoss()

        self.setup_flag = True

    def alternate_loss(self, logits, features, labels, net):

        return uniform_ce(logits)

    def reset(self, net, data_loader):
        """reset is done at each new "experiment" (dataset)

        """
        return

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

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        # print('*** postprocess on epoch', epoch)
        output = net(data)
        score = torch.softmax(output, dim=1)

        conf = torch.gather(score, -1, self.chunk_predicted_labels.unsqueeze(-1)).squeeze(-1)

        return self.chunk_predicted_labels, conf

    def finetune(self, net, data, epoch=0):
        """finetune is done after postprocess (that way you can
        retrieve results before any finetuning ; that's why
        postprocess is done epochs+1 times, to benefit from n=epochs
        finetuning)

        """

        beta = self.beta

        # for instance you can create a minibatch_loader
        minibatch_loader = DataLoader(list(zip(data, self.chunk_predicted_labels)), shuffle=True,
                                      batch_size=self.batch_size, drop_last=False)

        for mix_batch in minibatch_loader:

            try:
                id_batch = next(self.id_train_iter)

            except StopIteration:
                self.id_train_iter = iter(self.id_train_loader)
                id_batch = next(self.id_train_iter)

            id_data = id_batch['data'].cuda()
            id_label = id_batch['label'].cuda()
            mix_data, predicted_mix_labels = mix_batch

            id_output = net(id_data)
            mix_logits, mix_features = net(mix_data, return_feature=True)

            self.optimizer.zero_grad()

            loss = self.loss(id_output, id_label)
            loss += beta*self.alternate_loss(logits=mix_logits, features=mix_features,
                                             labels=predicted_mix_labels, net=net)

            loss.backward()
            self.optimizer.step()
