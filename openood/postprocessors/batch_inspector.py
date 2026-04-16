import numpy as np


class BatchInspector():

    _epoch = -1
    _iterations = {-1: 0}

    loss_ = {}
    weights_ = {}
    where_ = {}
    n_ = {}
    conf_ = {}

    def __init__(self, threshold=-2):
        self.threshold = threshold
        self.reset()

    @property
    def iterations(self):

        return self._iterations.get(self.epoch, 0)

    @iterations.setter
    def iterations(self, v):
        self._iterations[self.epoch] = v

    @property
    def epoch(self):
        return self._epoch

    @property
    def flushed(self):
        return not self.loss

    @epoch.setter
    def epoch(self, e):
        if e == self.epoch:
            # we are on the same epoch
            return
        # epoch chnage: tensor reset
        self.stats()
        self.reset()
        self._epoch = e
        self._iterations[e] = self._iterations[e - 1]

    def reset(self):
        self.loss = {}
        self.weights = {}
        self.where = np.array([])
        self.conf = np.array([])

    def quantiles(self, conf):

        a = [0.05, 0.5, 0.95]
        q = np.quantile(conf, a)
        q_ = dict(zip(q, a))
        q_[self.threshold] = (conf < self.threshold).mean()
        q_ = {_: len(conf) * q_[_] for _ in sorted(q_)}

        q_['mean'] = conf.mean()
        return q_

    def stats(self):
        e = self.epoch
        self.loss_[e] = {'raw': {}, 'weighted': {}, 'filtered': {},  'wf': {}}
        self.n_[e] = 0
        self.weights_[e] = {}

        wheres, wherecount = np.unique(self.where, return_counts=True)

        self.where_[e] = self.where

        if not self.loss:
            return
        n = len(self.loss['a'])
        self.n_[e] = n
        for _ in self.loss:

            self.loss_[e]['raw'][_] = self.loss[_].mean()

            weighted_loss = self.loss[_] * self.weights[_]
            self.loss_[e]['weighted'][_] = weighted_loss.mean()
            for w in wheres:
                self.loss_[e]['filtered']['{}_{}'.format(_, w)] = self.loss[_][self.where == w].mean()
                self.loss_[e]['wf']['{}_{}'.format(_, w)] = weighted_loss[self.where == w].sum() / n

        for _ in self.weights:
            self.weights_[e][_] = self.weights[_].mean()

        self.conf_[e] = {'all': self.quantiles(self.conf)}
        for _ in wheres:
            self.conf_[e][_] = self.quantiles(self.conf[self.where == _])

    def print(self):
        for e in range(self.epoch, -1, -1):
            n = self.n_.get(e)
            if n:
                break
        else:
            print('*** no epoch to sumup from {}'.format(self.epoch))
            return
        print('\n*** *** *** epoch {} chunk sumup (from epoch {}) *** *** ***'.format(e, self.epoch))
        for _ in self.loss_[e]:
            kept = [t[0] for t in self.loss_[e][_].items() if t[1] > 0]
            if len(kept) == 2:
                if kept[0].startswith('a'):
                    k_a = kept[0]
                    k_o = kept[1]
                else:
                    k_a = kept[1]
                    k_o = kept[0]
                self.loss_[e][_]['{}/{}'.format(k_a, k_o)] = self.loss_[e][_][k_a] / self.loss_[e][_][k_o]
            print('[chunk {} loss ({})]'.format(_, n), end=' ')
            print(' '.join('{}:{:.3g}'.format(*t) for t in self.loss_[e][_].items() if t[1] > 0))

        print('[chunk weights]', ' -- '.join('{}:{:.3g}'.format(*t)
                                             for t in self.weights_[e].items() if t[1] > 0))

        wheres, wherecount = np.unique(self.where_[e], return_counts=True)
        print('[chunk wheres]', ' -- '.join('{}: {}'.format(*_) for _ in zip(wheres, wherecount)))

        for w, q_ in self.conf_[e].items():
            m = q_.pop('mean', np.nan)
            print('[chunk conf {}]'.format(w),
                  ' || '.join('({:.0f}) < {:.2f}'.format(_[1], _[0]) for _ in q_.items()),
                  '-- m = {:.2f}'.format(m))

        print('[chunk it] {}'.format(self.iterations))

        self.n_[e] = 0
        print('*** *** ***\n')

    def update_mb(self, epoch, epochs, printout=False, **kw):
        self.epoch = epoch
        self.epochs = epochs

        if not kw:
            return

        self.iterations += 1

        loss = {}
        where = np.array(kw['where'])
        weights = kw['weights'].detach().cpu().numpy()
        weights = {'i': weights[:, 0], 'a': weights[:, 1]}
        conf = kw['conf'].detach().cpu().numpy()
        if 'stratified_loss_id' in kw:
            pass
            # strat_loss['o'] = kw['stratified_loss_id'].detach().cpu().numpy()
            # start_weights['o'] = np.ones_like(strat_loss['o'])
        else:
            loss['i'] = kw['id_loss'].detach().cpu().numpy()
        loss['a'] = kw['adaptation_loss'].detach().cpu().numpy()

        wheres, wherecount = np.unique(where, return_counts=True)
        if printout:
            print(' -- '.join('{}: {}'.format(*_) for _ in zip(wheres, wherecount)))

        for l in list(weights):
            for w in wheres:
                weights['{}_{}'.format(w, l)] = weights[l][where == w]
        for w in wheres:
            if printout:
                print('{}: {:.2f}, {:.2f}'.format(w, weights['{}_{}'.format(w, 'i')].mean(),
                                                  weights['{}_{}'.format(w, 'a')].mean()))

        w_means = {_: weights[_].mean() for _ in weights}
        w_means['r'] = w_means['i'] / w_means['a']
        if printout:
            print('[weights] i/a = {i:.2g}/{a:.2g}={r:.2g}'.format(**w_means))

        loss_means = {}
        for row, k in enumerate(loss):
            loss_means[k] = loss[k].mean()

        if 'stratified_loss_id' in kw:
            loss_means['o'] = kw['stratified_loss_id'].mean()

        loss_means['r'] = loss_means['a'] / loss_means['o']

        if printout:
            print('[loss] a/o = {a:.2g}/{o:.2g}={r:.2g}'.format(**loss_means))

        for _ in loss:
            self.loss[_] = np.hstack([self.loss.setdefault(_, np.array([])),
                                     loss[_]])
        for _ in weights:
            self.weights[_] = np.hstack([self.weights.setdefault(_, np.array([])),
                                        weights[_]])

        self.where = np.hstack([self.where, where])
        self.conf = np.hstack([self.conf, conf])
