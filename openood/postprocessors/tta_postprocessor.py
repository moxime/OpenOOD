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
from openood.recorders import get_recorder

from .base_postprocessor import BasePostprocessor

import torchvision


class PadBuffer(deque):

    def __init__(self, size, threshold, thr_key='conf', postprocessor=None):

        super().__init__(maxlen=size)
        self._thr_key = thr_key
        self.threshold = threshold
        self.postprocessor = postprocessor

    def append(self, **kw):

        assert self._thr_key in kw
        if kw[self._thr_key] <= self.threshold:
            super().append(kw)
            return 1
        return 0

    def empty(self):

        while True:
            try:
                self.pop()
            except IndexError:
                break

    def update_conf(self, net, batch_size=32, and_pred=False):
        if not self.postprocessor:
            return

        dl = DataLoader(self, batch_size=batch_size, drop_last=False, shuffle=False)

        b_ = []
        for b in dl:
            pred, conf = self.postprocessor.postprocess(net, b['data'], pred=None if and_pred else b['pred'])
            b['pred'] = pred
            b['conf'] = conf

            b = [dict(zip(b, t)) for t in zip(*b.values())]
            b_.extend(b)

        for old, new in zip(self, b_):
            old['pred'] = new['pred']
            old['conf'] = new['conf']

    def save(self, path):

        if self:
            torchvision.utils.save_image([_['data'] for _ in self], path)
            return path


class TTAPostprocessor(BasePostprocessor):

    """ TTAPostprocessor Loop

    - setup on net for a new "experiment"

    - for each dl of mix id/ood datasets do inference():

        - do reset() on net:

        - reload network

        - empty pad buffers

        - for each chunk of mix id/out samples

            - do new_chunk():

            - reload network (can be deactivated)

            - refresh pad buffers (in and ood)

            - for each epoch in 0...epochs-1

                - calculate conf score on chunk (if condition on epoch is
                  met with calculate_conf())

                - calculate conf score on ood pad

                - do init_epoch(epoch)

                - do finetume(epoch): see ft_tta_postprocessor for
                  implementation

            |____ end for eooch

            - calculate conf score on chunk

            - do init_epoch(last)

            - update self buffer

        |____ end for chink

    |____ end for dl """

    def __init__(self, config):
        super().__init__(config)

        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.checkpoint = config.network.checkpoint

        self.reload_network_at_chunk = config.postprocessor.reload_network

        self.chunk_size = self.config.pipeline.chunk_size

        pad_aux = [_ for _ in ('self', 'id', 'ood') if self.config.postprocessor.padding.get(_, 0) > 0]

        self.pad_sizes = {_: int(self.config.postprocessor.padding.get(_, 0) * self.chunk_size)
                          for _ in pad_aux}

        self.pad_thresholds = dict(self=-np.inf, id=np.inf, ood=np.inf)

        self_threshold = self.config.postprocessor.padding.threshold
        if self_threshold is not None:
            self.pad_thresholds['self'] = self_threshold

        self.debug = config.debug

        self.ood_ratio = config.pipeline.ood_ratio

        self.recorder = get_recorder(config)

    def padding_ood_filtering(self, net, id_ood_aux_loader_dict, n=10000, passes=20, buffer_length=4):

        assert 'ood' in self.aux_dls

        dl = self.aux_dls['ood']

        ratio = n / (passes * len(dl.dataset))
        n_keep = int(dl.batch_size * buffer_length * ratio)

        assert n_keep > 2

        kept_conf = (0., 0)

        for _ in range(passes):
            for __ in range(len(dl) // buffer_length):
                batch_ = [self.next_aux_batch('ood')['data'].to('cuda')
                          for __ in range(buffer_length)]

                """ self.postprocess(batch) returns pred, conf """
                conf = torch.hstack([self.postprocess(net, batch)[1] for batch in batch_])
                i_ = torch.sort(conf)[1][:n_keep]
                kept = torch.vstack(batch_)[i_]

                kept_conf = (kept_conf[0] + conf[i_].sum(), kept_conf[1] + len(i_))

            self.recorder.event('pad_ood_filter', dpass=_, n=kept_conf[1],
                                avge='{:.2f}'.format(kept_conf[0] / kept_conf[1]))

    def setup(self, net: nn.Module, id_loader_dict, id_ood_loader_dict):
        """setup is done once (for instance, get some metrics on the
        training id dataset

        """

        aux_dls = id_ood_loader_dict['aux']
        self.aux_dls = {_: aux_dls[_] for _ in aux_dls if _ in self.pad_sizes}
        self.pad_buffers = {_: PadBuffer(self.pad_sizes[_], self.pad_thresholds[_], postprocessor=self)
                            for _ in self.pad_sizes}

        def _unfold(name, dl):
            if isinstance(dl, DataLoader):
                self.recorder.event('dl', name, set='\n'+str(dl.dataset))
                return
            if not isinstance(dl, dict):
                return
            for k, v in dl.items():
                _unfold('{}/{}'.format(name, k), v)

        _unfold('id', id_loader_dict)
        _unfold('id_ood', id_ood_loader_dict)

        self.padding_ood_filtering(net, id_ood_loader_dict)

    def reload_network(self, net):
        net.load_state_dict(torch.load(self.checkpoint))
        self.recorder.event('reload_network')

    def reset(self, net):
        """reset is done at each new "experiment" (dataset)

        for instance, if postprocessor manages an history dict, this
        might be the place to do it (by overriding this method)

        """

        self.reload_network(net)

        for _ in self.pad_buffers:
            self.pad_buffers[_].empty()

    def next_aux_batch(self, where):
        """
         fetch next aux batch from aux_dl (recreate iter on stop iteration)
         """
        try:
            _ = next(self._aux_iters[where])
            self.recorder.event('aux_batch', where)
            return _
        except AttributeError:
            self._aux_iters = {}
        except (KeyError, StopIteration):
            self._aux_iters[where] = iter(self.aux_dls[where])

        return self.next_aux_batch(where)

    def update_pad_buffers(self, data, conf, pred, where='self', **kw):
        """ add x from data to pad_buffer[where], depending on conf and predicted label

        each element is added as a dictionary

        {'data': x, 'conf': conf, 'pred': pred, 'where': where}

        where can be 'ood' (known ood), 'id' (known id, e.g. from train) or 'self'

        """
        n = 0
        for x, s, y in zip(data, conf, pred):
            n += self.pad_buffers[where].append(data=x, pred=y, conf=s, where=where)

        self.recorder.event('update_pad_buffers', where, samples=n)
        return n

    def new_chunk(self, net, data, epochs=0):
        """
        done before postprocessing chunk (data)
        """
        if self.reload_network_at_chunk:
            self.reload_network(net)

        for _ in ('id', 'ood'):
            if _ not in self.pad_buffers:
                continue
            batch = self.next_aux_batch(_)
            data = batch['data'].cuda()
            if _ == 'id':
                pred = batch['label'].cuda()
                conf = torch.inf * torch.ones_like(pred)
            else:
                pred, conf = self.postprocess(net, data)

            self.update_pad_buffers(data, conf, pred, where=_)

    def calculate_conf(self, epoch=0, epochs=0):

        # when do we calculate conf
        return epoch in (0, epochs)

    def init_epoch(self, net, data, conf, pred, epoch=0, epochs=0):

        self.recorder.event('init_epoch', epoch=epoch)
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
                  epochs=0,
                  progress: bool = True):

        outputs_by_epochs = {_: {} for _ in ('pred', 'conf', 'label')}

        self.reset(net)

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
            self.new_chunk(net, data, epochs=epochs)

            """ Pred calculated here will the kept for all the FT

            """
            pred = None

            for epoch in range(epochs+1):

                """
                Important: we keep pred calculated prior to the FT
                """
                if self.calculate_conf(epoch=epoch, epochs=epochs):
                    self.recorder.event('calculate_conf', pred_is_none=pred is None)
                    pred, conf = self.postprocess(net, data, pred=pred)
                    for key, tensor in zip(('pred', 'conf', 'label'), (pred, conf, label)):
                        outputs_by_epochs[key].setdefault(epoch, []).append(tensor.cpu())

                    if self.pad_buffers.get('ood'):
                        self.pad_buffers['ood'].update_conf(net, self.batch_size, and_pred=True)

                self.init_epoch(net, data, conf, pred, epoch=epoch, epochs=epochs)

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
