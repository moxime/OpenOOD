import os
from contextlib import contextmanager
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
            self._reset_net_at_chunk = config.network.checkpoint
        else:
            self._reset_net_at_chunk = None

        self.debug = config.debug

    def add_to_progress_bar(self, data, conf, pred, num_chunk, epoch, epochs):

        n_aux = len(self.aux_set)
        r = (conf < 0).float().mean()
        q1 = conf.quantile(0.25)
        q2 = conf.quantile(0.5)
        q3 = conf.quantile(0.75)
        return (f'aux: {n_aux} [{epoch}/{epochs}] r={r:.1%}'
                f' conf: [{q1:.3f} -- {q2:.3f} -- {q3:.3f}]')

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """setup is done once (for instance, get some metrics on the
        training ind dataset

        """

        self.aux_set = []
        pass

    def reset(self, net, data_loader):
        """reset is done at each new "experiment" (dataset)

        for instance, if psotprocessor manages an history dict, this
        might be the place to do it (by overriding this method)

        """
        pass

    def update_aux_set(self, data, conf, pred, epoch=0, **kw):
        """ add x from data to aux_set, depending on conf and predicted label

        each element is added as a dictionary

        {'data': x, 'conf': conf, 'pred': pred, 'where': 'aux'}

        """
        pass

    def new_chunk(self, net, data):
        """
        done before postprocessing chunk (data)
        it will do the same that if you start with if epoch==0 in postprocess
        """

        if self._reset_net_at_chunk:
            net.load_state_dict(torch.load(self._reset_net_at_chunk))

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)

        return pred, conf

    @classmethod
    def _finetune_mode(cls, net, finetune=True):

        if isinstance(net, dict):
            for subnet in net.values():
                cls.finetune_mode(subnet, finetune=finetune)

        return
        for module in net.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                module.eval()
            else:
                module.train(finetune)

    @contextmanager
    def finetune_mode(self, net):
        self._finetune_mode(net, True)
        try:
            yield
        finally:
            self._finetune_mode(net, False)

    def finetune(self, net, data, conf, pred, epoch=0):
        """finetune is done after postprocess (that way you can
        retrieve results before any finetuning ; that's why
        postprocess is done epochs+1 times, to benefit from n=epochs
        finetuning)

        """

        # for instance you can create a minibatch_loader
        # ...
        pass

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  epochs=0,
                  progress: bool = True):

        outputs_by_epochs = {_: {} for _ in ('pred', 'conf', 'label')}

        self.reset(net, data_loader)

        progress_bar = tqdm(data_loader,
                            dynamic_ncols=True,
                            disable=not progress or not comm.is_main_process())
        num_chunk = 0
        for chunk in progress_bar:
            if (num_chunk > self.debug * len(data_loader)) and self.debug:
                break
            num_chunk += 1
            data = chunk['data'].cuda()
            label = chunk['label'].cuda()
            self.new_chunk(net, data)
            for epoch in range(epochs+1):

                pred, conf = self.postprocess(net, data, epoch=epoch)
                progress_bar.set_postfix_str(self.add_to_progress_bar(data,
                                                                      conf,
                                                                      pred,
                                                                      num_chunk,
                                                                      epoch,
                                                                      epochs))

                self.update_aux_set(data, conf, pred, epoch=epoch)

                if epoch < epochs:
                    with self.finetune_mode(net):
                        self.finetune(net, data, conf, pred, epoch=epoch)

                for key, tensor in zip(('pred', 'conf', 'label'), (pred, conf, label)):
                    outputs_by_epochs[key].setdefault(epoch, []).append(tensor.cpu())

            # convert values into numpy array
        for _ in outputs_by_epochs:
            for epoch in outputs_by_epochs[_]:
                outputs_by_epochs[_][epoch] = torch.cat(outputs_by_epochs[_][epoch]).numpy().astype(
                    float if _ == 'conf' else int)

        return tuple(outputs_by_epochs[_] for _ in ('pred', 'conf', 'label'))
