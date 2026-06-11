import pandas as pd
import openood.utils.comm as comm
from .info import num_classes_dict
from typing import Any

import numpy as np
import torch
import torch.nn as nn


import time


def timedfunc(message):
    def wrapper(func):
        def wrapped(obj, *a, **kw):
            t0 = time.time()
            out = func(obj, *a, **kw)
            if hasattr(obj, '_timefunc') and getattr(obj, '_timefunc'):
                print('{} executed in {:.2g}s'.format(message, time.time() - t0))
            return out
        return wrapped
    return wrapper


def _unfold_(obj, prefix='', prefix_inc='      '):

    if isinstance(obj, (list, tuple)):
        for _ in obj:
            _unfold_(obj, prefix=prefix+prefix_inc, prefix_inc=prefix_inc)

        return

    if isinstance(obj, dict):

        for _ in obj:
            print(prefix + _)
            _unfold_(obj[_], prefix=prefix+prefix_inc, prefix_inc=prefix_inc)

        return

    str_ = [type(obj)]

    if isinstance(obj, torch.utils.data.dataloader.DataLoader):
        str_l = '{N}= {n}x{m} s: {s} from '.format(n=len(obj), m=obj.batch_size,
                                                   N=len(obj) * obj.batch_size,
                                                   s=type(obj.sampler).__name__[0])
        d_str_ = str(obj.dataset).split('\n')

        str_ = [(' ' * len(str_l) if i else str_l) + _ for i, _ in enumerate(d_str_)]

    for _ in str_:
        print(prefix + _)


# A TTA PostProcessor to look at batches
def debug_tta(cls):

    def debugged(func):

        def debuggedfunc(obj, *a, **kw):
            obj._debug_now = obj._debug
            obj._debug = False
            out = func(obj, *a, **kw)
            obj._debug = obj._debug_now
            return out

        return debuggedfunc

    class DebugTTAPostprocessor(cls):
        def __init__(self, config):
            super().__init__(config)
            self.setup_flag = False
            self._debug = True
            self._timefunc = False
            self._epoch = -1

        @debugged
        def reload_network(self, net):

            super().reload_network(net)
            if self._debug_now:
                print('*** reloaded network')

        @debugged
        def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
            if self.setup_flag:
                return
            super().setup(net, id_loader_dict, ood_loader_dict)
            # self.loss = timedfunc('loss')(self.loss)

            if not self._debu_now:
                return
            print('ID LOADER DICT')
            _unfold_(id_loader_dict)

            print('OOD LOADER DICT')
            _unfold_(ood_loader_dict)

            for name, sub in net.named_children():

                assert (not any(_.requires_grad for _ in sub.parameters()) or
                        all(_.requires_grad for _ in sub.parameters()))

                print('****', name, any(_.requires_grad for _ in sub.parameters()))

            for _ in self.aux_dls:

                d = self.aux_dls[_].dataset
                print('*** aux_dl[{}]: {}'.format(_, d))

        @debugged
        def calculate_conf(self, epoch=0, epochs=0):

            b = super().calculate_conf(epoch=epoch, epochs=epochs)
            if b and self._debug_now:
                pass  # print('*** Calculating conf ({})'.format(epoch))
            return b

        @timedfunc('update pad buffers')
        @debugged
        def update_pad_buffers(self, data, conf, pred, where='self', **kw):

            n = super().update_pad_buffers(data, conf, pred, where=where, **kw)

            if self._debug_now:
                print('*** added {n}/{N} samples ({l}) to pad_{w}'.format(n=n,
                                                                          N=len(conf),
                                                                          l=len(self.pad_buffers[where]),
                                                                          w=where))

                print('*** x = {:.3f}+/-{:.3f}    [n={}]'.format(data.mean(), data.std(), len(data)))

            return n

        # @timedfunc('loss weight')
        @debugged
        def losses_weight(self, **kw):
            w = super().losses_weight(**kw)
            # if epoch in (epochs // 3, 2 * epochs // 3):
            #     print('{} {:4} conf={:.1f}: {:.1f}, {:.1f}'.format(epoch, where, conf, *w))
            return w

        @timedfunc('init epoch')
        @debugged
        def init_epoch(self, net, data, conf, pred, epoch=0, epochs=0):

            self._epoch = epoch
            if self.calculate_conf(epoch, epochs) and self._debug_now:
                print('*** epoch {} ***'.format(epoch))

                """ fc params """

                weight, bias = net.get_fc()

                print('*** b_y:', ' - '.join(map('{:.2f}'.format, bias)))
                print('*** ||my||²:', ' - '.join(map('{:.2f}'.format, (weight**2).sum(-1))))
                print('*** <my>:', ' - '.join(map('{:.2f}'.format, weight.mean(-1))))

            super().init_epoch(net, data, conf, pred, epoch=epoch, epochs=epochs)

        @timedfunc('post process')
        @debugged
        def postprocess(self, *a, **kw):
            pred, conf = super().postprocess(*a, **kw)

            if self._debug_now:
                t = self.pad_thresholds['self']
                a = [0.05, 0.5, 0.95]
                conf_ = conf.detach().cpu().numpy()
                q = np.quantile(conf_, a)
                q_ = dict(zip(q, a))
                q_[t] = (conf_ < t).mean()
                q_ = {_: len(conf_) * q_[_] for _ in sorted(q_)}

                print('[chunk ({}) calculated conf at epoch {}]'.format(len(conf_), self._epoch),
                      ' || '.join('({:.0f}) < {:.2f}'.format(_[1], _[0]) for _ in q_.items()))

            return pred, conf

        @timedfunc('adaptation loss')
        @debugged
        def adaptation_loss(self, *a, **kw):
            return super().adaptation_loss(*a, **kw)

        @timedfunc('FT')
        def finetune(self, *a, **kw):
            return super().finetune(*a, **kw)

        @timedfunc('next batch')
        def next_pad_batch(self, *a, **kw):
            return super().next_aux_batch(*a, **kw)

    return DebugTTAPostprocessor
