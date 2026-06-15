import os
import time
from pathlib import Path


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


class MBRecorder(dict):
    def update_minibatch(self, **kw):

        for k, v in kw.items():
            self.setdefault(k, []).append(v)


class BatchRecorder(list):

    def add_minibatch(self, mb, **kw):

        try:
            self[mb].update_minibatch(**kw)
        except IndexError:
            self.append(MBRecorder())
            self.add_minibatch(mb, **kw)


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

    def add_minibatch(self, mb, **kw):

        self.batch_recorder.add_minibatch(mb, **kw)


if __name__ == '__main__':

    class Config:

        def __init__(self, **kw):

            for k, w in kw.items():
                setattr(self, k, w)

    config = Config(recorder=Config(silent_events=[], print_events=True))
    recorder = TTARecorder(config)
