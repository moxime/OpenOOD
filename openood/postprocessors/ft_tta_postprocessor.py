import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from .tta_postprocessor import TTAPostprocessor


from openood.losses import uniform_ce, MarginCrossEntropy


class FTTTAPostprocessor(TTAPostprocessor):
    """ FTTTAPostprocessor(TTAPostprocessor)

    - implementation of finetume

    """

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

        for _ in self.stratified:
            self.aux_dls[_] = id_ood_loader_dict['aux'][_]

        self.loss = MarginCrossEntropy(margin=self.config.network.margin, reduction='none')
        self.optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.setup_flag = True

        unfreeze = self.config.postprocessor.ft.unfreeze
        self.recorder.event('unfreeze', unfreeze)
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

        return super().setup(net, id_loader_dict, id_ood_loader_dict)

    def adaptation_loss(self, logits, features, net):

        return uniform_ce(logits)

    def losses_weight(self, data=None, conf=0., pred=None, where='id', epoch=0, epochs=0):
        """ return loss_weight, adaptation_loss_weight for sample data

        where is either id, ood, self, mix

        """

        if where == 'id':
            return (1., 0.)

        return (0., self.beta)

    def finetune(self, net, data, conf, pred, epoch=0, epochs=0):
        """finetune is done  _epochs_ times

        """

        self.recorder.event('finetune', epoch='{}/{}'.format(epoch, epochs))
        self.recorder.ft_epoch('start', epoch, epochs, data=data, conf=conf, pred=pred)

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

        mb_generator = torch.Generator().manual_seed(0)
        mb_generator = None

        minibatch_loader = DataLoader(batch_list,
                                      shuffle=batch_list,  # if batch_list is empty, do not shuffle (bug)
                                      batch_size=self.batch_size,
                                      generator=mb_generator,
                                      drop_last=False)  # TBT: dop_last=True

        if epoch == 0:
            # to track clipped_grad (see below)
            self._clipped_grad = 0
            self._grad = 0

        grad_norm = 0.
        for i, batch in enumerate(minibatch_loader):

            data = batch['data'].cuda()
            pred = batch['pred'].cuda()
            conf = batch['conf'].cuda()
            where = batch['where']

            weights = batch['weights'].cuda()
            w_norm0 = weights.norm(0, dim=0)

            if any(conf.isnan()):
                raise ValueError('{} NaN in conf'.format(conf.isnan().int().sum()))

            logits, features = net(data, return_feature=True)

            if w_norm0[0] or True:
                id_loss = self.loss(logits, pred)
            else:
                id_loss = torch.zeros_like(conf)

            if w_norm0[1] or True:
                adaptation_loss = self.adaptation_loss(logits=logits, features=features, net=net)
            else:
                adaptation_loss = torch.zeros_like(conf)

            self.recorder.add_minibatch('batch', i, where=where,
                                        conf=conf,
                                        id_loss=id_loss,
                                        id_weights=w_normalization[0] * weights.T[0],
                                        adaptation_loss=adaptation_loss,
                                        adaptation_weights=w_normalization[1] * weights.T[1])

            self.optimizer.zero_grad()

            # weights is batch_size*2
            loss = w_normalization[0] * (id_loss * weights.T[0]).mean()
            loss += w_normalization[1] * (adaptation_loss * weights.T[1]).mean()

            for _ in self.stratified:
                batch_ = self.next_aux_batch(_)
                data = batch_['data'].cuda()
                logits, features = net(data, return_feature=True)
                if _ == 'ood':
                    w = self.beta
                    stratified_loss = self.adaptation_loss(logits, features, net)
                elif _ == 'id':
                    w = 1.
                    pred = batch_['label'].cuda()
                    stratified_loss = self.loss(logits, pred)

                self.recorder.add_minibatch('strat', i, **{'stratified_{}'.format(_): stratified_loss})
                loss += (w * stratified_loss).mean()

            loss.backward()

            grad_norm_i = torch.nn.utils.clip_grad_norm_(net.parameters(), 200)
            grad_norm += grad_norm_i
            self.recorder.event('grad_norm_i', '{}-{} {:.3g}'.format(epoch, i, grad_norm_i),
                                loss='{:.3g}'.format(loss))
            if grad_norm > 200:
                self._clipped_grad += 1
            self._grad += 1
            self.optimizer.step()

        self.recorder.event('grad_norm', '{}: {:.3g}'.format(epoch, grad_norm))

        self.recorder.ft_epoch('end', epoch, epochs)
