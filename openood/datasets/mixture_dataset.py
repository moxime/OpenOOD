from contextlib import contextmanager
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler, SequentialSampler, BatchSampler


class MixtureDataset(Dataset):

    _flattens_sub = True

    @classmethod
    @contextmanager
    def flatten_sub(cls, b=True):

        old_value = cls._flattens_sub
        cls._flattens_sub = b

        try:
            yield

        finally:
            cls._flattens_sub = old_value

    def __init__(self, **datasets):

        super().__init__()

        self._lengths = {_: len(datasets[_]) for _ in datasets}
        self._datasets = dict(**datasets)
        self._len = sum(len(_) for _ in datasets.values())

        self._cumlengths = {}

        i = 0
        for d in datasets:
            self._cumlengths[i] = d
            i += len(datasets[d])

    def __len__(self):

        return self._len

    def __getitem_from_tuple__(self, t):

        def flattens(t1, t2):

            if not isinstance(t2, tuple):
                return (t1, t2)

            return tuple((t1, *flattens(*t2)))

        sub, i = t

        # if i >= self._lengths[sub]:
        #     raise IndexError('{}[{}>{}]'.format(sub, i, self._lengths[sub]))

        data, label = self._datasets[sub][i]

        if self._flattens_sub:
            return data, flattens(sub, label)
        else:
            return data, (sub, label)

    def __getitem__(self, i):

        if isinstance(i, tuple):

            return self.__getitem_from_tuple__(i)

        for start in reversed(self._cumlengths):
            if i >= start:
                break

        subset = self._cumlengths[start]

        return self.__getitem_from_tuple__((subset, i - start))

    def __repr__(self):

        s_d = {_: repr(self._datasets[_]) for _ in self._datasets}

        for sub in s_d:

            s = '\n'.join('{} {}'.format(sub if i == 0 else ' '*(len(sub)-2)+'|-', line)
                          for i, line in enumerate(s_d[sub].split('\n')))
            s_d[sub] = s

        return '\n'.join(s_d[_] for _ in s_d)
        # return '\n'.join(['{}: {}'.format(_, repr(self._datasets[_])) for _ in self._datasets])


class MixtureTimeSampler(Sampler):

    def __init__(self, data_source, period=np.inf, SubSampler=RandomSampler, **kw):

        try:
            lengths = data_source._lengths

        except AttributeError:
            lengths = {'__self__': len(data_source)}

        self.lengths = lengths

        self._len = sum(lengths.values())

        self._first_index = {}
        i = 0
        for d in lengths:
            self._first_index[d] = i
            i += lengths[d]

        self._subsamplers = {d: SubSampler(range(l)) for d, l in lengths.items()}

        self._period = period

    def __len__(self):
        return self._len

    def __iter__(self):

        iters = {_: iter(self._subsamplers[_]) for _ in self._subsamplers}
        already_chosen = {_: 0 for _ in self.lengths}

        choose_from = list(already_chosen)

        chosen_idx = None

        a = 1/self._period

        for t in range(len(self)):
            p_ = [(self.lengths[_] - already_chosen[_]) / (len(self)-t) for _ in iters]
            probabilites = torch.tensor(p_)
            # print('===', t)
            # print(already_chosen, ' '.join(map('{:.4f}'.format, p_)))

            if chosen_idx is not None and probabilites[chosen_idx] > 0:
                probabilites *= a
                probabilites[chosen_idx] += (1 - a)

            chosen_idx = torch.multinomial(probabilites, 1)

            chosen = choose_from[chosen_idx]
            already_chosen[chosen] += 1

            # print('[{:d}]    '.format(*chosen_idx), ' '.join(map('{:.4f}'.format, probabilites)))
            index = next(iters[chosen]) + self._first_index[chosen]
            yield index


class IDOODDataset(MixtureDataset):

    def __init__(self, ind, ood):

        super().__init__(ind=ind, ood=ood)


class IDOODSampler(Sampler):

    def __init__(self, ind_sampler, ood_sampler, ood_rate=0.1, **kw):
        """Params

        - ind_sampler is a infinite random sampler on ind dataset

        _ ood_sampler is a random sampler on ood

        """

        assert ood_rate <= 1
        assert ood_rate * len(ind_sampler) >= (1 - ood_rate) * len(ood_sampler)

        self.ood_rate = ood_rate
        self.samplers = dict(ind=ind_sampler, ood=ood_sampler)

        self._len = int(len(ood_sampler) / ood_rate)

    def __iter__(self):

        iters = {_: iter(self.samplers[_]) for _ in self.samplers}
        for _ in range(len(self)):
            if torch.rand(1) <= self.ood_rate:
                try:
                    yield ('ood', next(iters['ood']))
                except StopIteration:
                    break
            else:
                yield ('ind', next(iters['ind']))

    def __len__(self):

        return self._len


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from torchvision.datasets import MNIST

    mnist_train = MNIST(root='~/dev/dnn/torch-data/')

    class FakeDataset(Dataset):

        def __init__(self, prefix, len=10000, num_of_classes=10):

            self._proto = prefix + '{:05d}'
            self._len = len
            self.num_of_classes = num_of_classes

        def __getitem__(self, i):

            if i < 0:
                return self[i + len(self)]

            if i >= self._len:
                raise IndexError('{}>={}'.format(i, self._len))
            return self._proto.format(i), i % self.num_of_classes

        def __len__(self):

            return self._len

        def __repr__(self):
            return '[{}..{}]'.format(self[0][0], self[-1][0])

    def create_datasets(oods=4, N=10000, ood_prefix='ood-'):
        d_names = list('abcdefghijklmnopqrstuvwxyz01234567890')[:oods]
        oodsets = {_: FakeDataset(ood_prefix+_, N) for _ in d_names}

        oodset = MixtureDataset(**oodsets)

        indset = MixtureDataset(x=FakeDataset('ind--', N))

        ind_ood_set = IDOODDataset(indset, oodset)

        return indset, oodset, ind_ood_set

    def test_mixture(K=4, N=10000):

        _, d, _ = create_datasets(oods=K, N=N)

        # del m_

        try:
            old_m_ = m_
        except NameError:
            old_m_ = None

        m_ = [1000]

        m_ = [1, 100, 500, 1000, 5000, 10000]

        if old_m_ != m_:
            streams = {}

        for m in m_:
            if old_m_ == m_:
                break

            alpha = min(len(d_names) / (len(d_names) - 1) / m, 1)
            p_0 = 1e-102 + alpha * (1-1/len(d_names))
            m_mean_th = 1/p_0
            m_std_th = (1-p_0)**0.5/p_0

            s = MixtureTimeSampler(d, period=m)

            loader = DataLoader(d, sampler=s, batch_size=1)

            current_subset = None
            stream_lengths = []
            for batch in loader:
                for subset, data in zip(*batch):
                    if subset != current_subset:
                        current_subset = subset
                        stream_lengths.append(1)
                    else:
                        stream_lengths[-1] += 1

            stream_lengths = np.array(stream_lengths)
            streams[m] = stream_lengths

            print('m = {:8}: {:.1f} +-{:.1f} ({:.1f}+-({:.1f})'.format(m,
                                                                       stream_lengths.mean(),
                                                                       stream_lengths.std(),
                                                                       m_mean_th, m_std_th))

        figs = {}
        plt.close('all')
        for m in streams:

            fig_name = '{}'.format(m)
            figs[fig_name] = plt.figure(fig_name)

            stream = streams[m]

            ax = figs[fig_name].gca()
            ax.clear()

            ax.plot(stream.cumsum(), streams[m], 'x')

            mean = streams[m].mean()
            std = streams[m].std()

            ax.set_title('{:.1f} +/- {:.1f}'.format(mean, std))

            ax.hlines([mean - std, mean, mean+std], 0, K * N, 'r', 'dashed')
            ax.hlines([mean], 0, K * N, 'r')
            # for v, f in ((-std, 'r--'), (0, 'r'), (std, '--r')):
            #     ax.plot(stream.cumsum(), mean * np.ones_like(stream) + v, f)

            lengths, counts = np.unique(stream, return_counts=True)

            counts *= lengths

            fig_name += ' hist'
            figs[fig_name] = plt.figure(fig_name)

            ax = figs[fig_name].gca()

            ax.bar(lengths, counts)

        for f in figs:
            if 'hist' in f:
                figs[f].show()

    def test_ind_ood_sampler(N=1000, ood_rate=0.1, n_oods=4, batch_size=512, period=1e5):

        indset, oodset, in_ood_set = create_datasets(oods=n_oods, N=N)

        ind_sampler = RandomSampler(indset, num_samples=50*len(indset))
        ood_sampler = MixtureTimeSampler(oodset,  period=period)

        idod_sampler = IDOODSampler(ind_sampler, ood_sampler, ood_rate=ood_rate)

        num = dict(ind=0, ood=0)
        for i, _ in enumerate(idod_sampler):
            num[_[0]] += 1

        n = i + 1
        print('{} samples drawn {:.1%} of which are ood'.format(n, num['ood']/n))

        batch_sampler = BatchSampler(idod_sampler, batch_size=512, drop_last=True)

        loader = DataLoader(in_ood_set, batch_size=batch_size,
                            sampler=idod_sampler)

        stats = {'ood-'+_: [] for _ in oodset._datasets}
        stats['ind-x'] = []

        for _, (iod, s, y) in loader:

            for sub in stats:
                stats[sub].append(sum(_ == sub.split('-')[1] for _ in s) / len(y))

        return dset, loader, stats

    plt.close('all')
    for period in [1, 2, 10, 100, 1e5]:
        dset, loader, stats = test_ind_ood_sampler(N=10000, ood_rate=0.5, n_oods=2, period=period)

        fig = plt.figure('P={}'.format(period))
        for _ in stats:

            fig.gca().plot(stats[_], label=_)
        fig.legend()

        fig.show()
