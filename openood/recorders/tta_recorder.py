import torch
import numpy as np
import pandas as pd


class Events(dict):

    def __init__(self, silent, white_list=[], black_list=[], **kw):

        self.silent = silent
        self.black_list = black_list
        self.white_list = white_list
        super().__init__(**kw)

    def __getitem__(self, k):

        return super().get(k, k)

    def __contains__(self, k):
        if k in self.white_list:
            return True
        if self.silent:
            return False
        if k in self.black_list:
            return False

        return True


class BatchRecorder(dict):

    stats = {}

    def add_minibatch(self, df_k, i, split_keys=True, **kw):

        if df_k not in self:
            self[df_k] = pd.DataFrame()

        for k, v in kw.items():
            if isinstance(v, torch.Tensor):
                kw[k] = v.detach().cpu()
        df = pd.DataFrame(kw)

        df['mb'] = i
        df.index.name = 'sample'
        df.set_index('mb', inplace=True, append=True)
        df.index = df.index.swaplevel(1, 0)

        if 'where' in kw:
            df.set_index('where', inplace=True, append=True)

        if split_keys:
            cols = [_.split('_') for _ in df.columns]
            nlevels = max(len(_) for _ in cols)
            for _ in cols:
                while len(_) < nlevels:
                    _.insert(0, '_')

                df.columns = pd.MultiIndex.from_tuples(cols)

        self[df_k] = pd.concat([self[df_k], df]).groupby(df.index.names).max()

        self._compute_stats()

    def _compute_stats(self):

        self.stats = {}

        for df_k, df in self.items():
            for c in df.columns:
                if c[1] == 'loss' and (c[0], 'weights') in df.columns:
                    df[(c[0], 'weighted_loss')] = df[(c[0], 'loss')] * df[(c[0], 'weights')]

            if 'where' in df.index.names:
                self.stats[df_k] = df.groupby('where').mean()
                self.stats['batch'].loc[df_k] = df.mean()
                df_counts = df.groupby('where').count()
                df_counts = df_counts[df_counts.columns[0]]
                df_counts.loc['batch'] = df_counts.sum()
                self.stats[df_k]['count'] = df_counts
            else:
                df_stat = df.mean()
                if isinstance(df_stat, pd.Series):
                    df_stat = pd.DataFrame(df_stat).T
                    df_stat.index = ['batch']
                self.stats[df_k] = df_stat


class TTARecorder:

    _events_str = {'reload_network': 'Reload Network'}

    def __init__(self, config):

        self.config = config

        black_list = config.recorder.get('event_blacklist', [])
        white_list = config.recorder.get('event_whitelist', [])
        self.events = Events(not config.recorder.print_events,
                             black_list=black_list, white_list=white_list,
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
            for _ in self.batch_recorder.stats:
                df = self.batch_recorder.stats[_]
                self.event('mb_stat', '{}\n{}'.format(_,
                                                      df.to_string(float_format='{:.3g}'.format)))

    def add_minibatch(self, *a, **kw):

        self.batch_recorder.add_minibatch(*a, **kw)


if __name__ == '__main__':

    class Config:

        def __init__(self, **kw):

            for k, w in kw.items():
                setattr(self, k, w)

        def get(self, k, o=None):

            try:
                return getattr(self, k)
            except AttributeError:
                return o

    config = Config(recorder=Config(event_blacklist=[], event_whitelist=[], print_events=True))
    recorder = TTARecorder(config)

    def fake(n, dtype=float):

        if dtype is str:
            return np.random.choice(['mix', 'pad'], n)
        return np.random.randn(n)

    n = 4
    recorder.ft_epoch('start', 0, 1)
    for mb in range(3):

        where = fake(n, str)
        split_keys = False
        split_keys = True
        recorder.add_minibatch('batch', mb, id_loss=fake(n), id_weights=fake(n),
                               adaptation_loss=fake(n), adaptation_weights=fake(n),
                               conf=fake(n),
                               where=where,
                               split_keys=split_keys)

        recorder.add_minibatch('strat', mb, id_loss=fake(n),
                               split_keys=split_keys)

        recorder.add_minibatch('strat', mb, pad_loss=fake(n),
                               split_keys=split_keys)

        # recorder.add_minibatch('batch', mb, conf=fake(n), where=where,
        #                        split_keys=split_keys)

    recorder.ft_epoch('end', 0, 1)
