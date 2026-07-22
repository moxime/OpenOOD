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

        def flatten(*seq):
            # TBF
            for e in seq:
                if isinstance(e, (list, tuple)):
                    yield from flatten(*e)
                else:
                    yield e

        sub, i = t

        sample = self._datasets[sub][i]

        if isinstance(sample, dict):
            label = sample['label']
        else:
            label = sample[1]

        if self._flattens_sub:
            label = tuple(flatten(sub, label))
        else:
            label = (sub, label)

        if isinstance(sample, dict):
            sample['label'] = label
        else:
            sample = sample[0], label

        sample.pop('soft_label', None)
        return sample

    def __getitem__(self, i):

        if isinstance(i, int) and i < 0:

            return self.__getitem__(i + self._len)

        if isinstance(i, tuple):

            return self.__getitem_from_tuple__(i)

        for start in reversed(self._cumlengths):
            if i >= start:
                break

        subset = self._cumlengths[start]

        return self.__getitem_from_tuple__((subset, i - start))

    def __repr__(self, header=''):

        elbow = "└──"
        pipe = "│  "
        tee = "├──"
        blank = "   "
        str_list = []
        for i, (name, d) in enumerate(self._datasets.items()):
            last = i == len(self._datasets) - 1
            if not isinstance(d, MixtureDataset):
                d_str_ = repr(d).split('\n')
                for ii, s in enumerate(d_str_):
                    if ii:
                        str_list.append(header + (blank if last else pipe) + ' ' * len(name + '  ') + s)

                    else:
                        str_list.append(header + (elbow if last else tee) + '{}: {}'.format(name, s))

            else:
                str_list.append(header + (elbow if last else tee) + name)
                child_str_list = d.__repr__(header=header + (blank if last else pipe)).split('\n')
                str_list.extend(child_str_list)

        return '\n'.join(str_list)


class MixtureTimeSampler(Sampler):

    def __init__(self, data_source, period=np.inf, SubSampler=RandomSampler, **kw):

        try:
            lengths = data_source._lengths

        except AttributeError:
            lengths = {'__self__': len(data_source)}

        self.lengths = lengths

        self._unit_length = min(lengths.values())

        self._len = self._unit_length * len(lengths)

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

        DEBUG = False

        iters = {_: iter(self._subsamplers[_]) for _ in self._subsamplers}
        remainders = {_: self._unit_length for _ in self.lengths}

        choose_from = list(remainders)

        chosen_idx = None
        chosen = None

        a = 1/self._period

        for t in range(len(self)):

            remain_current_subset = remainders.get(chosen, 0)
            change_subset = torch.rand(1) < a

            # if no remaining in current subset, change subset
            if remain_current_subset == 0:
                change_subset = 1

            # all remaining samples are in current subset
            if remain_current_subset == len(self) - t:
                change_subset = 0

            change_subset = (((torch.rand(1) < a) or (remain_current_subset == 0)) and
                             (remain_current_subset < len(self) - t))

            if DEBUG:
                print('*** a:', a, 't:', t, '/', len(self),  'remain prev:',
                      remain_current_subset, len(self) - t - remain_current_subset,
                      chosen, remainders)

            if change_subset:
                p_ = [remainders[_] / (len(self) - t - remain_current_subset) for _ in iters]
                if chosen_idx is not None:
                    p_[chosen_idx] = 0.
                probabilites = torch.tensor(p_)
                chosen_idx = torch.multinomial(probabilites, 1)

                if DEBUG:
                    print(' '.join(map('{:.2e}'.format, probabilites.numpy())))

            chosen = choose_from[chosen_idx]
            remainders[chosen] -= 1
            # print(chosen, remainders, remain_prev)

            # print('[{:d}]    '.format(*chosen_idx), ' '.join(map('{:.4f}'.format, probabilites)))
            index = next(iters[chosen]) + self._first_index[chosen]
            yield index


class IDOODDataset(MixtureDataset):

    def __init__(self, ind, **oods):

        ood = MixtureDataset(**oods)

        self.sub_ood = {-i-1: _ for i, _ in enumerate(oods)}
        self._ood_idx = {_: -i-1 for i, _ in enumerate(oods)}

        super().__init__(ind=ind, ood=ood)

    def __getitem__(self, i):
        """get item

        label given by super() is in the form of (gt, sub, label)

            - gt is ind / ood

            - sub is dataset

            - label is the class

        here we want the label to be

            - label if gt is ind
            - -ood_idx (negative label) if gt is ood_conf

        """

        if isinstance(i, int) and i < 0:
            return self.__getitem__(i + len(self))

        sample = super().__getitem__(i)

        # labels is gt, sub, [<sub/sub>,] label
        labels = sample['label'] if isinstance(sample, dict) else sample[1]

        gt = labels[0]
        sub = labels[1]

        label = labels[-1]

        if gt == 'ood':
            label = self._ood_idx[sub]

        print('***', labels, label)

        if isinstance(sample, dict):
            sample['label'] = label
        else:
            sample = sample[0], label

        return sample


class IDOODSampler(Sampler):

    def __init__(self, data_source, ood_ratio=0.1, ood_period=1,
                 IDSampler=RandomSampler, OODSubsampler=RandomSampler, **kw):
        """Params

        - data_source has to be a IDOODSampler

        """

        assert ood_ratio <= 1
        assert isinstance(data_source, IDOODDataset)

        ind_sampler = IDSampler(data_source._datasets['ind'], num_samples=int(1e7))
        ood_sampler = MixtureTimeSampler(data_source._datasets['ood'],  period=ood_period,
                                         SubSampler=OODSubsampler)

        assert ood_ratio * len(ind_sampler) >= (1 - ood_ratio) * len(ood_sampler)

        self.ood_ratio = ood_ratio
        self.samplers = dict(ind=ind_sampler, ood=ood_sampler)

        self._len = int(len(ood_sampler) / ood_ratio)

    def __iter__(self):

        iters = {_: iter(self.samplers[_]) for _ in self.samplers}
        for _ in range(len(self)):
            if torch.rand(1) <= self.ood_ratio:
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
    import sys

    mnist_train = MNIST(root='~/dev/dnn/torch-data/')

    class FakeDataset(Dataset):

        def __init__(self, prefix, len=10000, num_of_classes=10, dtype='tuple'):

            self._proto = prefix + '{:05d}'
            self._len = len
            self.num_of_classes = num_of_classes
            self._dtype = dtype

        def __getitem__(self, i):

            if i < 0:
                return self[i + len(self)]

            if i >= self._len:
                raise IndexError('{}>={}'.format(i, self._len))

            data, label = self._proto.format(i), i % self.num_of_classes
            if self._dtype == 'tuple':
                return data, label

            return {'data': data, 'label': label}

        def __len__(self):

            return self._len

        def __repr__(self):
            if self._dtype == 'tuple':
                first, last = self[0][0], self[-1][0]
            else:
                first, last = self[0]['data'], self[-1]['data']
            return '[{}..{}]\n[FAKE]\n[TR]'.format(first, last)

    def create_datasets(oods=4, N=10000, ood_prefix='ood-', dtype='tuple'):
        d_names = list('abcdefghijklmnopqrstuvwxyz01234567890')[:oods]
        oodsets = {_: FakeDataset(ood_prefix+_, (i+1) * N, dtype=dtype)
                   for i, _ in enumerate(d_names)}

        oodset = MixtureDataset(**oodsets)

        indset = MixtureDataset(x=FakeDataset('ind--', N, dtype=dtype))

        ind_ood_set = IDOODDataset(indset, **oodsets)

        return indset, oodset, ind_ood_set

    ind, ood, ind_ood_set = create_datasets()

    print(ind_ood_set)

    sys.exit()

    def test_mixture(K=4, N=10000, period=1e5, batch_size=50, fig=True, dtype='dict'):

        _, d, _ = create_datasets(oods=K, N=N, dtype=dtype)

        alpha = min(K / (K - 1) / period, 1)
        p_0 = 1e-102 + alpha * (1-1/K)
        m_mean_th = 1/p_0
        m_std_th = (1-p_0)**0.5/p_0

        s = MixtureTimeSampler(d, period=period)

        loader = DataLoader(d, sampler=s, batch_size=batch_size)

        current_subset = None
        stream = []
        entropy = []
        for batch in loader:
            if isinstance(batch, tuple):
                data, labels = batch
            else:
                data, labels = batch['data'], batch['label']

            _, (s_, y) = data, labels
            value, counts = np.unique(s_, return_counts=True)
            entropy.append(np.log2(len(s_)) - (counts * np.log2(counts)).sum() / len(s_))
            for subset in s_:
                if subset != current_subset:
                    current_subset = subset
                    stream.append(1)
                else:
                    stream[-1] += 1

        stream = np.array(stream)
        entropy = np.array(entropy).mean()

        print('m = {:8}: {:5.1e} +-{:5.1e} ({:5.1e}+-({:5.1e}) 2^H={:.2f}'.format(period,
                                                                                  stream.mean(),
                                                                                  stream.std(),
                                                                                  m_mean_th,
                                                                                  m_std_th,
                                                                                  2**entropy))

        if fig:
            fig_name = '{}'.format(period)
            fig = plt.figure(fig_name)

            ax = fig.gca()

            ax.clear()
            ax.plot(stream.cumsum(), stream, 'x')

            mean = stream.mean()
            std = stream.std()

            ax.set_title('{:.1f} +/- {:.1f}'.format(mean, std))

            ax.hlines([mean - std, mean, mean+std], 0, K * N, 'r', 'dashed')
            ax.hlines([mean], 0, K * N, 'r')
            # for v, f in ((-std, 'r--'), (0, 'r'), (std, '--r')):
            #     ax.plot(stream.cumsum(), mean * np.ones_like(stream) + v, f)

            lengths, counts = np.unique(stream, return_counts=True)

            counts *= lengths

            fig.show()

            fig_name += ' hist'
            fig = plt.figure(fig_name)

            ax = fig.gca()

            ax.bar(lengths, counts)

            fig.show()

        # returns last batch
        return stream, entropy, s_

    def test_ind_ood_sampler(N=1000, ood_ratio=0.1, n_oods=4, batch_size=512, period=np.inf,
                             dtype='dict', fig=True):

        indset, oodset, in_out_set = create_datasets(oods=n_oods, N=N, dtype=dtype)

        in_out_sampler = IDOODSampler(in_out_set, ood_ratio=ood_ratio, ood_period=period)

        num = dict(ind=0, ood=0)
        for i, _ in enumerate(in_out_sampler):
            num[_[0]] += 1

        n = i + 1
        print('{} samples drawn, {} ({:.1%}) of which are ood'.format(n, num['ood'], num['ood']/n))

        loader = DataLoader(in_out_set, batch_size=batch_size,
                            sampler=in_out_sampler)

        stats = {'ood-'+_: [] for _ in oodset._datasets}
        stats['ind-x'] = []

        for batch in loader:
            if isinstance(batch, tuple):
                label = batch[1]
            else:
                label = batch['label']

            s = [in_out_set.sub_ood.get(int(_), 'x') for _ in label]
            for sub in stats:
                stats[sub].append(sum(_ == sub.split('-')[1] for _ in s) / len(s))

        if fig:
            fig = plt.figure('P={}'.format(period))
            y_ = np.vstack(list(stats.values()))
            x = list(range(len(y_[0])))
            fig.gca().stackplot(x, y_, labels=list(stats))

            fig.legend()
            fig.show()

        return in_out_set, loader, stats

    plt.close('all')

    N = 10000

    p_ = [1, 2, 10, 100, 1000, 10000, 1e500]
    p_ = [1e4]
    p_ = [1, 100, 1000, 10000, np.inf]
    p_ = []

    if p_:
        stream, entropy, batch = zip(*(test_mixture(K=4, N=N, period=period, fig=True)
                                       for period in p_))

    p_ = [1, 2, 10, 100, 1000, 10000, 1e500]
    p_ = [10, 20, 50, 100, 200, 500]
    p_ = [1]
    p_ = []
    p_ = [1,  np.inf]
    p_ = [1, 100, 1000, 10000, np.inf]

    if p_:
        dset, loader, stats = zip(*(test_ind_ood_sampler(N=N, ood_ratio=0.1, n_oods=4,
                                                         period=period, fig=True)
                                    for period in p_))

    if sys.argv[0]:
        input()
