import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from .tta_postprocessor import TTAPostprocessor

from .batch_inspector import BatchInspector

from openood.losses import uniform_ce, MarginCrossEntropy


class FTTTAPostprocessor(TTAPostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.batch_size = self.config.ood_dataset.batch_size

        self.stratified = [_ for _ in ('id', 'ood') if self.config.postprocessor.padding.get(_, 0) < 0]

        self.setup_flag = False
        self.ft_args = self.config.postprocessor.ft
        self.lr = self.ft_args.lr
        self.wd = self.ft_args.wd
        self.beta = self.ft_args.beta

        self.epochs = self.ft_args.epochs

        self.filterout_null_weights = self.ft_args.only_non_zeros
        self.size_normalization = True

        print(f"*** params lr={self.lr} beta={self.beta} self thr={self.pad_thresholds['self']}")

    def setup(self, net: nn.Module, id_loader_dict, id_ood_loader_dict):
        if self.setup_flag:
            return
        super().setup(net, id_loader_dict, id_ood_loader_dict)

        for _ in self.stratified:
            self.aux_dls[_] = id_ood_loader_dict['aux'][_]

        self.loss = MarginCrossEntropy(margin=self.config.network.margin, reduction='none')
        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.setup_flag = True

        unfreeze = self.config.postprocessor.ft.unfreeze
        if unfreeze:
            for _ in net.parameters():
                _.requires_grad_(False)

            for _ in net.get_fc_layer().parameters():
                _.requires_grad_(True)

        if unfreeze in ('all', 'penultimate'):
            for name, submodule in reversed(list(net.named_children())):
                for p in submodule.parameters():
                    p.requires_grad_(True)

                if name.lower().startswith('layer') and unfreeze == 'penultimate':
                    break

    def next_aux_minibatch(self, where):

        assert where in self.stratified, '{} is ot stratified'.format(where)
        try:
            return next(self._aux_iters[where])
        except AttributeError:
            self._aux_iters = {}
        except (KeyError, StopIteration):
            self._aux_iters[where] = iter(self.aux_dls[where])

        return self.next_aux_minibatch(where)

    def adaptation_loss(self, logits, features, net):

        return uniform_ce(logits)

    def losses_weight(self, data=None, conf=0., pred=None, where='id', epoch=0, epochs=0):
        """ return loss_weight, adaptation_loss_weight for sample data

        where is either id, ood, self, mix

        """

        if where == 'id':
            return (1., 0.)

        return (0., self.beta)

    def inspect_minibatch(self, epoch=0, epochs=0, flush=False, **kw):
        """
        implement this methd in child class for debug purpose
        """
        if not hasattr(self, '_debug'):
            return

        for _ in self.pad_buffers:
            self.pad_buffers[_].save(os.path.join(self.config.exp_name, 'pad_{}.png'.format(_)))

        if not hasattr(self, '_inspector'):
            self._inspector = BatchInspector(threshold=self.pad_thresholds.get('self', -2))

        self._inspector.epoch = epoch

        printout = (not epoch or self.calculate_conf(epoch+1, epochs)) and flush
        printout = flush
        self._inspector.update_mb(epoch, epochs=epochs, printout=printout, **kw)
        if printout:
            self._inspector.print()

    def finetune(self, net, data, conf, pred, epoch=0, epochs=0):
        """finetune is done  _epochs_ times

        """

        mix_batch = {'conf': conf, 'pred': pred, 'data': data, 'where': ['mix' for _ in pred]}
        batch_list = [dict(zip(mix_batch, t)) for t in zip(*mix_batch.values())]

        for where in self.pad_buffers:
            batch_list.extend(self.pad_buffers[where])

        for d in batch_list:
            d.pop('weights', None)
            d['weights'] = torch.tensor(self.losses_weight(**d, epoch=epoch, epochs=epochs))

        if self.filterout_null_weights:
            batch_list = [_ for _ in batch_list if _['weights'].norm()]

        if self.size_normalization:
            N_samples = len(batch_list)
            N_id_weights = len([_ for _ in batch_list if _['weights'][0] > 0])
            N_adaptation_weights = len([_ for _ in batch_list if _['weights'][1] > 0])

            w_normalization = (N_samples / (N_id_weights + 1e-12),
                               N_samples / (N_adaptation_weights + 1e-12))

        else:
            w_normalization = (1., 1.)

        minibatch_loader = DataLoader(batch_list,
                                      shuffle=batch_list,  # if batch_list is empty, do not shuffle (bug)
                                      batch_size=self.batch_size,
                                      drop_last=False)

        if epoch == 0:
            # to track clipped_grad (removed, see below)
            self._clipped_grad = 0
            self._grad = 0

        for i, batch in enumerate(minibatch_loader):

            inspection_dict = {}

            data = batch['data'].cuda()
            pred = batch['pred'].cuda()
            conf = batch['conf'].cuda()
            where = batch['where']

            weights = batch['weights'].cuda()
            w_norm0 = weights.norm(0, dim=0)

            inspection_dict.update(where=where, conf=conf)

            if any(conf.isnan()):
                raise ValueError('{} NaN in conf'.format(conf.isnan().int().sum()))

            logits, features = net(data, return_feature=True)

            if w_norm0[0]:
                id_loss = self.loss(logits, pred)
            else:
                id_loss = torch.zeros_like(conf)

            if w_norm0[1]:
                adaptation_loss = self.adaptation_loss(logits=logits, features=features, net=net)
            else:
                adaptation_loss = torch.zeros_like(conf)

            inspection_dict.update(id_loss=id_loss, adaptation_loss=adaptation_loss, weights=weights)

            self.optimizer.zero_grad()

            # weights is batch_size*2
            loss = w_normalization[0] * (id_loss * weights.T[0]).mean()
            loss += w_normalization[1] * (adaptation_loss * weights.T[1]).mean()

            for _ in self.stratified:
                batch_ = self.next_aux_minibatch(_)
                data = batch_['data'].cuda()
                logits, features = net(data, return_feature=True)
                if _ == 'ood':
                    w = self.beta
                    stratified_loss = self.adaptation_loss(logits, features, net)
                elif _ == 'id':
                    w = 1
                    pred = batch_['label'].cuda()
                    stratified_loss = self.loss(logits, pred)

                inspection_dict.update({'stratified_loss_{}'.format(_): stratified_loss})
                loss += (w * stratified_loss).mean()

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 200)
            if grad_norm > 200:
                self._clipped_grad += 1
            self._grad += 1
            self.optimizer.step()

            self.inspect_minibatch(**inspection_dict, epoch=epoch, epochs=epochs)

        self.inspect_minibatch(epoch=epoch, epochs=epochs, flush=True)
