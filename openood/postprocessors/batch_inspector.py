import numpy as np


class BatchInspector():

    epoch = -1

    def empty(self):
        self.loss = {}
        self.loss_w = {}
        self.where = np.array([])

    def means(self):

        for _ in self.loss:
            self.loss[_] = self.loss[_].mean()

        for _ in self.loss_w:
            self.loss_w[_] = self.loss_w[_].mean()

    def print(self):
        print('\n*** *** *** epoch {} chunk sumup *** *** ***'.format(self.epoch))

        self.loss['r'] = self.loss['a'] / self.loss['o']
        print('[chunk loss] a/o = {a:.3g}/{o:.3g}={r:.3g}'.format(**self.loss))
        print('[chunk weights]', ' -- '.join('{}:{:.3g}'.format(*t)
                                             for t in self.loss_w.items()))

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
        w_loss = kw['w_loss'].detach().cpu().numpy()
        if 'stratified_loss_id' in kw:
            loss['o'] = kw['stratified_loss_id'].detach().cpu().numpy()
        else:
            loss['o'] = kw['original_loss'].detach().cpu().numpy() * w_loss[0]
        loss['a'] = kw['alternate_loss'].detach().cpu().numpy() * w_loss[1]

        wheres, wherecount = np.unique(where, return_counts=True)
        print(' -- '.join('{}: {}'.format(*_) for _ in zip(wheres, wherecount)))

        loss_w = {}
        for w in wheres:
            for i, l in enumerate('oa'):
                loss_w['{}_{}'.format(w, l)] = w_loss[i, where == w]

            print('{}: {:.2f}, {:.2f}'.format(w, loss_w['{}_{}'.format(w, 'o')].mean(),
                                              loss_w['{}_{}'.format(w, 'a')].mean()))

        w_means = dict(o=w_loss.mean(-1)[0], a=w_loss.mean(-1)[1])
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
        for _ in loss_w:
            self.loss_w[_] = np.hstack([self.loss_w.setdefault(_, np.array([])),
                                        loss_w[_]])

        self.where = np.hstack([self.where, where])
