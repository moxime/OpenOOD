import os
from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm

from .base_postprocessor import BasePostprocessor


class TTAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)

        # batch_size for fine_tuning
        self.batch_size = self.config.ood_dataset.batch_size
        try:
            self.batch_size = self.config.postprocessor.batch_size
        except AttributeError:
            pass

        try:
            self.reset_net_at_chunk = self.config.postprocessor.reset_network
        except AttributeError:
            self.reset_net_at_chunk = True

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """setup is done once (for instance, get some metrics on the
        training ind dataset

        """
        if self.reset_net_at_chunk:
            self.reset_net_at_chunk = os.path.join(self.config.output_dir, 'tmp.ckpt')
            net.save(self.reset_net_at_chunk)

    def reset(self, net, data_loader):
        """reset is done at each new "experiment" (dataset)

        for instance, if psotprocessor manages an history dict, this
        might be the place to do it (by overriding this method)

        """
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        if epoch == 0 and self.reset_net_at_chunk:
            net.load(self.reset_net_at_chunk)

        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)

        if epoch == 0:
            self.predicted_labels = pred

        return pred, conf

    def finetune(self, net, data, epoch=0):
        """finetune is done after postprocess (that way you can
        retrieve results before any finetuning ; that's why
        postprocess is done epochs+1 times, to benefit from n=epochs
        finetuning)

        - minibatch_loader is a shuffled datalpder on the processed
        chunk with batch_size=self.batch_size

        """

        # for instance you can create a minibatch_loader
        minibatch_loader = DataLoader(data, shuffle=True, batch_size=self.batch_size)
        # ...
        pass

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  epoch=0,
                  epochs=0,
                  progress: bool = True):

        lists = {_: {} for _ in ('pred', 'conf', 'label')}

        self.reset(net, data_loader)
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            for epoch in range(epochs+1):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                pred, conf = self.postprocess(net, data, epoch=epoch)
                self.finetune(net, data, epoch=epoch)

                for key, tensor in zip(('pred', 'conf', 'label'), (pred, conf, label)):
                    lists[key].setdefault(epoch, []).append(tensor.cpu())

            # convert values into numpy array
        for _ in lists:
            for epoch in lists[_]:
                lists[_][epoch] = torch.cat(lists[_][epoch]).numpy().astype(
                    float if _ == 'conf' else int)

        return tuple(lists[_] for _ in ('pred', 'conf', 'label'))
