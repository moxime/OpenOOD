import numpy as np


class BatchInspector():

    _epoch = -1
    _iterations = {-1: 0}

    loss = {}
    weights = {}
    where = np.array([])

    @property
    def iterations(self):

        return self._iterations.get(self.epoch, 0)

    @iterations.setter
    def iterations(self, v):
        self._iterations[self.epoch] = v

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, e):
        if e == self.epoch:
            # we are on the same epoch
            return
        # epoch chnage: tensor reset
        self.reset()
        self._epoch = e
        self._iterations[e] = self._iterations[e - 1]

    def flush(self):
        self.means()
        self.print()

    def reset(self):
        self.loss = {}
        self.weights = {}
        self.where = np.array([])

    def means(self):
        self.loss_ = {'raw': {}, 'weighted': {}, 'filtered': {},  'wf': {}}
        wheres, wherecount = np.unique(self.where, return_counts=True)

        for _ in self.loss:
            self.loss_['raw'][_] = self.loss[_].mean()

            weighted_loss = self.loss[_] * self.weights[_]
            self.loss_['weighted'][_] = weighted_loss.mean()
            for w in wheres:
                self.loss_['filtered']['{}_{}'.format(_, w)] = self.loss[_][self.where == w].mean()
                n = len(self.loss['a'])
                self.loss_['wf']['{}_{}'.format(_, w)] = weighted_loss[self.where == w].sum() / n

        for _ in self.weights:
            self.weights[_] = self.weights[_].mean()

    def print(self):
        print('\n*** *** *** epoch {} chunk sumup *** *** ***'.format(self.epoch))
        n = len(self.loss['a'])
        for _ in self.loss_:
            kept = [t[0] for t in self.loss_[_].items() if t[1] > 0]
            if len(kept) == 2:
                if kept[0].startswith('a'):
                    k_a = kept[0]
                    k_o = kept[1]
                else:
                    k_a = kept[1]
                    k_o = kept[0]
                self.loss_[_]['{}/{}'.format(k_a, k_o)] = self.loss_[_][k_a] / self.loss_[_][k_o]
            print('[chunk {} loss ({})]'.format(_, n), end=' ')
            print(' '.join('{}:{:.3g}'.format(*t) for t in self.loss_[_].items() if t[1] > 0))

        print('[chunk weights]', ' -- '.join('{}:{:.3g}'.format(*t)
                                             for t in self.weights.items() if t[1] > 0))

        wheres, wherecount = np.unique(self.where, return_counts=True)
        print('[chunk wheres]', ' -- '.join('{}: {}'.format(*_) for _ in zip(wheres, wherecount)))

        print('[chunk it] {}'.format(self.iterations))

        print('*** *** ***\n')

    def update_mb(self, epoch, epochs, flush=False, **kw):
        self.epoch = epoch
        self.epochs = epochs

        if not kw:
            if flush:
                self.flush()
            return

        loss = {}
        where = np.array(kw['where'])
        weights = kw['weights'].detach().cpu().numpy()
        weights = {'i': weights[:, 0], 'a': weights[:, 1]}
        if 'stratified_loss_id' in kw:
            pass
            # strat_loss['o'] = kw['stratified_loss_id'].detach().cpu().numpy()
            # start_weights['o'] = np.ones_like(strat_loss['o'])
        else:
            loss['i'] = kw['id_loss'].detach().cpu().numpy()
        loss['a'] = kw['adaptation_loss'].detach().cpu().numpy()

        wheres, wherecount = np.unique(where, return_counts=True)
        print(' -- '.join('{}: {}'.format(*_) for _ in zip(wheres, wherecount)))

        for l in list(weights):
            for w in wheres:
                weights['{}_{}'.format(w, l)] = weights[l][where == w]
        for w in wheres:
            print('{}: {:.2f}, {:.2f}'.format(w, weights['{}_{}'.format(w, 'i')].mean(),
                                              weights['{}_{}'.format(w, 'a')].mean()))

        w_means = {_: weights[_].mean() for _ in weights}
        w_means['r'] = w_means['i'] / w_means['a']
        print('[weights] i/a = {i:.2g}/{a:.2g}={r:.2g}'.format(**w_means))

        loss_means = {}
        for row, k in enumerate(loss):
            loss_means[k] = loss[k].mean()

        if 'stratified_loss_id' in kw:
            loss_means['o'] = kw['stratified_loss_id'].mean()

        loss_means['r'] = loss_means['a'] / loss_means['o']

        print('[loss] a/o = {a:.2g}/{o:.2g}={r:.2g}'.format(**loss_means))

        for _ in loss:
            self.loss[_] = np.hstack([self.loss.setdefault(_, np.array([])),
                                      loss[_]])
        for _ in weights:
            self.weights[_] = np.hstack([self.weights.setdefault(_, np.array([])),
                                         weights[_]])

        self.where = np.hstack([self.where, where])

        if flush:
            self.flush()
