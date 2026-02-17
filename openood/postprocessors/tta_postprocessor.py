import os
import numpy as np
from contextlib import contextmanager
from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm

from .base_postprocessor import BasePostprocessor

import time


class TTAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)

        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        # batch_size for fine_tuning
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

    def setup(self, net: nn.Module, id_loader_dict, id_ood_loader_dict):
        """setup is done once (for instance, get some metrics on the
        training id dataset

        """
        pass

    def reload_network(self, net):
        net.load_state_dict(torch.load(self.checkpoint))

    def reset(self, net, data_loader, padding_id_dl=None, padding_ood_dl=None):
        """reset is done at each new "experiment" (dataset)

        for instance, if postprocessor manages an history dict, this
        might be the place to do it (by overriding this method)

        """

        self.reload_network(net)

        self.pad_sets = {_: [] for _ in ('self', 'ood', 'id')}

        self.pad_dls = {}
        if padding_id_dl:
            self.pad_dls['id'] = padding_id_dl
        if padding_ood_dl:
            self.pad_dls['ood'] = padding_ood_dl
        self.pad_iters = {_: iter(d) for _, d in self.pad_dls.items()}

    def update_pad_set(self, data, conf, pred, where='self', **kw):
        """ add x from data to pad_set, depending on conf and predicted label

        each element is added as a dictionary

        {'data': x, 'conf': conf, 'pred': pred, 'where': where}

        where can be ood (knonw ood from ext_padding), id (known id eg from train) or self

        """
        n = 0
        threshold = self.pad_thresholds[where]

        for x, s, y in zip(data, conf, pred):

            if s <= threshold:
                self.pad_sets[where].append(dict(data=x, pred=y, conf=s, where=where))
                n += 1
                if len(self.pad_sets[where]) > self.pad_sizes[where]:
                    self.pad_sets[where].pop(0)
        return n

    def new_chunk(self, net, data):
        """
        done before postprocessing chunk (data)
        """

        if self.reload_network_at_chunk:
            self.reload_network(net)

        for p in self.pad_dls:
            """
            id_batch = {_: id_batch[_].cuda() for _ in ('label', 'data')}
            id_batch['pred'] = id_batch.pop('label')
            id_batch['where'] = ['id' for _ in id_batch['pred']]
            id_batch['conf'] = torch.tensor([np.inf for _ in id_batch['pred']]).cuda()
            """
            try:
                batch = next(self.pad_iters[p])
            except StopIteration:
                self.pad_iters[p] = iter(self.pad_dls[p])
                batch = next(self.pad_iters[p])

            data = batch['data'].cuda()

            if p == 'id':
                pred = batch['label'].cuda()
                conf = torch.tensor([np.inf for _ in pred]).cuda()

            else:
                pred, conf = self.postprocess(net, data)

            n = self.update_pad_set(data, conf, pred, where=p)

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

        n_pad = len(self.pad_sets['self'])
        r = (conf < 0).float().mean()

        min_conf = conf.min()
        max_conf = conf.max()
        q1 = conf.quantile(0.25)
        q2 = conf.quantile(0.5)
        q3 = conf.quantile(0.75)

        s = (f'self pad: {n_pad} (+{delta_pad}) [{epoch}/{epochs}]'
             f' conf: < {min_conf:6.3f} --[{q1:6.3f} | {q2:6.3f} | {q3:6.3f}]-- {max_conf:5.3f} >')

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

        self.reset(net, data_loader, padding_id_dl=padding_id_dl, padding_ood_dl=padding_ood_dl)

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
                pred, conf = self.postprocess(net, data, pred=pred)

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

            delta_self_pad = self.update_pad_set(data=data,
                                                 conf=conf,
                                                 pred=pred,
                                                 where='self')
        # convert values into numpy array
        for _ in outputs_by_epochs:
            for epoch in outputs_by_epochs[_]:
                outputs_by_epochs[_][epoch] = torch.cat(outputs_by_epochs[_][epoch]).numpy().astype(
                    float if _ == 'conf' else int)

        return tuple(outputs_by_epochs[_] for _ in ('pred', 'conf', 'label'))
