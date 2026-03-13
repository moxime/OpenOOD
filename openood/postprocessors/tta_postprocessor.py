import os
import numpy as np
from collections import deque
from contextlib import contextmanager
from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm

from .base_postprocessor import BasePostprocessor


class PadBuffer(deque):

    def __init__(self, size, threshold, thr_key='conf'):

        super().__init__(maxlen=size)
        self._thr_key = thr_key
        self.threshold = threshold

    def append(self, **kw):

        assert self._thr_key in kw
        if kw[self._thr_key] <= self.threshold:
            super().append(kw)
            return 1
        return 0


class TTAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)

        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.checkpoint = config.network.checkpoint

        self.reload_network_at_chunk = config.postprocessor.reload_network

        self.chunk_size = self.config.pipeline.chunk_size

        self.pad_sizes = {_: int(self.config.postprocessor.padding.get(_, 0) * self.chunk_size)
                          for _ in ('self', 'id', 'ood')}

        self.pad_thresholds = dict(self=-np.inf, id=np.inf, ood=np.inf)

        self_threshold = self.config.postprocessor.padding.threshold
        if self_threshold is not None:
            self.pad_thresholds['self'] = self_threshold

        self.debug = config.debug

        self.ood_ratio = config.pipeline.ood_ratio

    def setup(self, net: nn.Module, id_loader_dict, id_ood_loader_dict):
        """setup is done once (for instance, get some metrics on the
        training id dataset

        """
        pass

    def reload_network(self, net):
        net.load_state_dict(torch.load(self.checkpoint))

    def reset(self, net, data_loader, aux_id_dl=None, aux_ood_dl=None):
        """reset is done at each new "experiment" (dataset)

        for instance, if postprocessor manages an history dict, this
        might be the place to do it (by overriding this method)

        """

        self.reload_network(net)

        self.pad_buffers = {_: PadBuffer(self.pad_sizes[_], self.pad_thresholds[_])
                            for _ in ('self', 'ood', 'id')}

        self.pad_dls = {}
        if aux_id_dl and self.pad_sizes['id'] > 0:
            self.pad_dls['id'] = aux_id_dl
        if aux_ood_dl and self.pad_sizes['ood'] > 0:
            self.pad_dls['ood'] = aux_ood_dl

    def next_pad_batch(self, where):

        assert where in self.pad_dls, where

        try:
            return next(self._pad_iters[where])
        except AttributeError:
            self._pad_iters = {}
        except (KeyError, StopIteration):
            self._pad_iters[where] = iter(self.pad_dls[where])

        return self.next_pad_batch(where)

    def update_pad_buffers(self, data, conf, pred, where='self', **kw):
        """ add x from data to pad_set, depending on conf and predicted label

        each element is added as a dictionary

        {'data': x, 'conf': conf, 'pred': pred, 'where': where}

        where can be 'ood' (knonw ood from ext_padding), 'id' (known id eg from train) or 'self'

        """
        n = 0
        if where in self.stratified:
            return 0
        for x, s, y in zip(data, conf, pred):
            n += self.pad_buffers[where].append(data=x, pred=y, conf=s, where=where)

        return n

    def new_chunk(self, net, data):
        """
        done before postprocessing chunk (data)
        """
        if self.reload_network_at_chunk:
            self.reload_network(net)

    def calculate_conf(self, epoch=0, epochs=0):

        # when do we calculate conf
        return epoch in (0, epochs)

    def init_epoch(self, net, data, conf, pred, epoch=0, epochs=0):

        return

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, pred=None):
        """ postprocess is done for each "chunk" of data at each epoch
        """
        output = net(data)
        score = torch.softmax(output, dim=1)

        if pred is None:
            _, pred = torch.max(score, dim=1)
        conf = torch.gather(score, -1, pred.unsqueeze(-1)).squeeze(-1)

        return pred, conf

    @classmethod
    def _finetune_mode(cls, net, finetune=True):

        if isinstance(net, dict):
            for subnet in net.values():
                cls._finetune_mode(subnet, finetune=finetune)

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

    def finetune(self, net, data, conf, pred, epoch=0, epochs=0):
        """finetune is done after postprocess (that way you can
        retrieve results before any finetuning ; that's why
        postprocess is done epochs+1 times, to benefit from n=epochs
        finetuning)

        """

        # for instance you can create a minibatch_loader
        # ...
        pass

    def epoch_sumup(self, data, conf, pred, delta_pad, num_chunk, epoch, epochs):

        n_pad = len(self.pad_buffers['self'])
        r = (conf < 0).float().mean()

        min_conf = conf.min()
        max_conf = conf.max()
        q1 = conf.quantile(self.ood_ratio)
        q2 = conf.quantile(0.5)
        q3 = conf.quantile(0.75)

        s = (f'self pad: {n_pad} (+{delta_pad}) [{epoch}/{epochs}]'
             f' conf: < {min_conf:5.2f} --[{q1:5.2f} | {q2:5.2f} | {q3:5.2f}]-- {max_conf:4.2f} >')

        try:
            s = '\{}\ '.format(self._clipped_grad) + s
        except AttributeError:
            pass

        return s

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  padding_id_dl=None,
                  padding_ood_dl=None,
                  epochs=0,
                  progress: bool = True):

        outputs_by_epochs = {_: {} for _ in ('pred', 'conf', 'label')}

        self.reset(net, data_loader, aux_id_dl=padding_id_dl, aux_ood_dl=padding_ood_dl)

        progress_bar = tqdm(data_loader,
                            dynamic_ncols=True,
                            disable=not progress or not comm.is_main_process())
        num_chunk = 0
        delta_self_pad = '--'
        for chunk in progress_bar:
            if (num_chunk > self.debug * len(data_loader)) and self.debug:
                break
            num_chunk += 1
            data = chunk['data'].cuda()
            label = chunk['label'].cuda()
            self.new_chunk(net, data)

            """ Pred calculated here will the kept for all the FT

            """
            pred = None

            for epoch in range(epochs+1):

                """
                Important: we keep pred calculated prior to the FT
                """
                if self.calculate_conf(epoch=epoch, epochs=epochs):
                    pred, conf = self.postprocess(net, data, pred=pred)

                self.init_epoch(net, data, conf, pred, epoch=epoch, epochs=epochs)

                for key, tensor in zip(('pred', 'conf', 'label'), (pred, conf, label)):
                    outputs_by_epochs[key].setdefault(epoch, []).append(tensor.cpu())

                if epoch < epochs:
                    with self.finetune_mode(net):
                        self.finetune(net, data, conf, pred, epoch=epoch, epochs=epochs)

                progress_bar.set_postfix_str(self.epoch_sumup(data,
                                                              conf,
                                                              pred,
                                                              delta_self_pad,
                                                              num_chunk,
                                                              epoch,
                                                              epochs))

            delta_self_pad = self.update_pad_buffers(data=data,
                                                     conf=conf,
                                                     pred=pred,
                                                     where='self')
        # convert values into numpy array
        for _ in outputs_by_epochs:
            for epoch in outputs_by_epochs[_]:
                outputs_by_epochs[_][epoch] = torch.cat(outputs_by_epochs[_][epoch]).numpy().astype(
                    float if _ == 'conf' else int)

        return tuple(outputs_by_epochs[_] for _ in ('pred', 'conf', 'label'))
