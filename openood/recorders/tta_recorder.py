import os
import time
from pathlib import Path
import torch
import numpy as np


class Events(dict):

    def __init__(self, silent, *black_list, **kw):

        self.silent = silent
        self.black_list = black_list
        super().__init__(**kw)

    def __getitem__(self, k):

        return super().get(k, k)

    def __contains__(self, k):
        if self.silent:
            return False
        if k in self.black_list:
            return False

        return True


class BatchRecorder(dict):

    _stats = {}

    def add_minibatch(self, mb, **kw):
        for k, v in kw.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()

            v = np.asarray(v)
            t = (v.shape[-1], v.sum(-1), (v**2).sum(-1))
            self.setdefault(k, []).append(t)

        self._compute_stats()

    def _compute_stats(self):

        self._stats['mean'] = dict()
        self._stats['std'] = dict()

        for k, mbs in self.items():

            mean = 0
            mean_square = 0
            n = 0
            for _ in mbs:
                n += _[0]
                mean += _[1]
                mean_square += _[2]

            mean /= n
            mean_square /= n
            self._stats['mean'][k] = mean
            self._stats['std'][k] = np.sqrt(mean_square - mean**2)


class TTARecorder:

    _events_str = {'reload_network': 'Reload Network'}

    def __init__(self, config):

        self.config = config

        black_list = [] if not config.recorder.silent_events else config.recorder.silent_events
        self.events = Events(not config.recorder.print_events, *black_list,
                             **self._events_str)

    def event(self, evt, *a, **kw):

        if evt not in self.events:
            return
        print('***', self.events[evt], *a, ' '.join('{}:{}'.format(k, w) for k, w in kw.items()))

    def ft_epoch(self, action, epoch, epochs, **kw):
        if action == 'start':
            self.batch_recorder = BatchRecorder()
            return

        if action == 'end':
            for _ in self.batch_recorder._stats:
                print('***', _)
                print(self.batch_recorder._stats[_])

    def add_minibatch(self, mb, **kw):

        self.batch_recorder.add_minibatch(mb, **kw)


if __name__ == '__main__':

    class Config:

        def __init__(self, **kw):

            for k, w in kw.items():
                setattr(self, k, w)

    config = Config(recorder=Config(silent_events=[], print_events=True))
    recorder = TTARecorder(config)
