import numpy as np


class BatchInspector():

    epoch = -1

    def empty(self):
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
        for _ in self.loss_:
            if 'a' in self.loss_[_]:
                self.loss_[_]['a/o'] = self.loss_[_]['a'] / self.loss_[_]['o']
            print('[chunk {} loss]'.format(_), end=' ')
            print(' '.join('{}:{:.3g}'.format(*t) for t in self.loss_[_].items() if t[1] > 0))

        print('[chunk weights]', ' -- '.join('{}:{:.3g}'.format(*t)
                                             for t in self.weights.items() if t[1] > 0))

        wheres, wherecount = np.unique(self.where, return_counts=True)
        print('[chunk wheres]', ' -- '.join('{}: {}'.format(*_) for _ in zip(wheres, wherecount)))

        print('*** *** ***\n')

    def update_mb(self, epoch, **kw):
        if epoch != self.epoch:
            if self.epoch > -1:
                self.means()
                self.print()
            self.empty()
        self.epoch = epoch

        loss = {}
        where = np.array(kw['where'])
        weights = kw['weights'].detach().cpu().numpy()
        weights = {'o': weights[0], 'a': weights[1]}
        if 'stratified_loss_id' in kw:
            loss['o'] = kw['stratified_loss_id'].detach().cpu().numpy()
        else:
            loss['o'] = kw['original_loss'].detach().cpu().numpy()
        loss['a'] = kw['alternate_loss'].detach().cpu().numpy()

        wheres, wherecount = np.unique(where, return_counts=True)
        print(' -- '.join('{}: {}'.format(*_) for _ in zip(wheres, wherecount)))

        for l in list(weights):
            for w in wheres:
                weights['{}_{}'.format(w, l)] = weights[l][where == w]
        for w in wheres:
            print('{}: {:.2f}, {:.2f}'.format(w, weights['{}_{}'.format(w, 'o')].mean(),
                                              weights['{}_{}'.format(w, 'a')].mean()))

        w_means = dict(o=weights['o'].mean(), a=weights['a'].mean())
        w_means['r'] = w_means['o'] / w_means['a']
        print('[weights] orig/alt = {o:.2g}/{a:.2g}={r:.2g}'.format(**w_means))

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
