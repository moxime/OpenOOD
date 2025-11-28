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
        self.batch_size = self.config.postprocessor.get('batch_size',
                                                        self.config.ood_dataset.batch_size)

        if self.config.postprocessor.get('reset_network', True):
            self.reset_net_at_chunk = config.network.checkpoint
        else:
            self.reset_net_at_chunk = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """setup is done once (for instance, get some metrics on the
        training ind dataset

        """
        pass

    def reset(self, net, data_loader):
        """reset is done at each new "experiment" (dataset)

        for instance, if psotprocessor manages an history dict, this
        might be the place to do it (by overriding this method)

        """
        pass

    def new_chunk(self, net, data):
        """
        done before postprocessing chunk (data)
        it will do the same that if you start with if epoch==0 in postprocess
        """

        if self.reset_net_at_chunk:
            net.load_state_dict(torch.load(self.reset_net_at_chunk))

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)

        return pred, conf

    def finetune(self, net, data, epoch=0):
        """finetune is done after postprocess (that way you can
        retrieve results before any finetuning ; that's why
        postprocess is done epochs+1 times, to benefit from n=epochs
        finetuning)

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
                  progress_bar=None,
                  progress: bool = True):

        outputs_by_epochs = {_: {} for _ in ('pred', 'conf', 'label')}

        self.reset(net, data_loader)
        num_chunk = 0

        if progress_bar is None:
            progress_bar = tqdm(data_loader,
                                dynamic_ncols=True,
                                disable=not progress or not comm.is_main_process())
        for batch in progress_bar:
            num_chunk += 1
            if num_chunk == 10:
                break
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            self.new_chunk(net, data)
            for epoch in range(epochs+1):
                progress_bar.set_postfix_str('{}:{}'.format(num_chunk, epoch))

                pred, conf = self.postprocess(net, data, epoch=epoch)
                if epoch < epochs:
                    self.finetune(net, data, epoch=epoch)

                for key, tensor in zip(('pred', 'conf', 'label'), (pred, conf, label)):
                    outputs_by_epochs[key].setdefault(epoch, []).append(tensor.cpu())

            # convert values into numpy array
        for _ in outputs_by_epochs:
            for epoch in outputs_by_epochs[_]:
                outputs_by_epochs[_][epoch] = torch.cat(outputs_by_epochs[_][epoch]).numpy().astype(
                    float if _ == 'conf' else int)

        return tuple(outputs_by_epochs[_] for _ in ('pred', 'conf', 'label'))
