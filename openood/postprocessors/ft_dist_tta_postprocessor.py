import os
from pathlib import Path
from typing import Any
import yaml
import numpy as np
import torch
import torch.nn as nn
from .ft_tta_postprocessor import FTTTAPostprocessor


class DistTTAPostprocessor(FTTTAPostprocessor):

    def __init__(self, config):

        super().__init__(config)
        assert not self.args or self.args.mu_ood in ('zero', 'mean', None)
        self.mu_ood = self.args.mu_ood if self.args else None
        self.iterations = self.args.iterations_per_phase
        self.switch_phase = self.epochs // 4

        self.ft_checkpoint = None
        if self.args.gasless:
            self.ft_checkpoint = Path(config.output_dir) / 'checkpoint_ft.pth'

        # number of iterations per phase when no self padding
        min_padded_size = self.chunk_size + sum(self.pad_sizes.values()) - self.pad_sizes.get('self', 0)
        min_it_per_epoch = min_padded_size / self.batch_size
        self.iterations_per_phase = int(min_it_per_epoch * self.switch_phase)
        self.max_iterations = self.args.max_iterations_per_phase
        self.ft_args.iterations_per_phase = self.iterations_per_phase
        if self.max_iterations:
            assert not self.pad_sizes.get('id', 0)

        print('*** mu_ood', self.mu_ood, 'iter/phase {} {}'.format('=' if self.max_iterations else '>=',
                                                                   self.iterations_per_phase))

        config_save_path = os.path.join(config.output_dir, 'config.yml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config,
                      f,
                      default_flow_style=False,
                      sort_keys=False,
                      indent=2)

    def setup(self, net: nn.Module, id_loader_dict, id_ood_loader_dict):

        if self.mu_ood:
            if self.mu_ood == 'zero':
                self.mu_ood = torch.zeros_like(net.get_fc_layer().weight[0])
            else:
                self.mu_ood = net.get_fc_layer().weight.detach().mean(0)

        return super().setup(net, id_loader_dict, id_ood_loader_dict)

    def reset(self, *a, **kw):

        super().reset(*a, **kw)

        self.ft_checkpoint_loaded = False
        if self.ft_checkpoint:
            try:
                os.remove(self.ft_checkpoint)
                self.recorder.event('load_ft_state', rm=self.ft_checkpoint.name)
            except FileNotFoundError:
                self.recorder.event('load_ft_state', rm='{} does not exist'.format(self.ft_checkpoint.name))

    def epoch_sumup(self, *a, **kw):
        return '[{}]'.format(self.phase[:3]) + super().epoch_sumup(*a, **kw)

    def adaptation_loss(self, logits, features, net):

        return features.square().sum(-1)

    def losses_weight(self, conf=0., where='id', epoch=0, epochs=0, **kw):
        """ return id_loss_weight, adaptation_loss_weight for sample x

        if mix :

        - during first half (epoch < epoch / 4): keep every sample

        - during second half: only if likely ood (conf < sb) """

        if self.phase == 'solid':
            # No FT in solid phase
            return (0., 0.)

        if where == 'mix':
            if self.phase == 'gas':
                # first phase
                return (0., self.beta)
            # second phase (liquid)
            if conf < self.pad_thresholds['self']:
                return (0., self.beta)
            return (0., 0.)

        return super().losses_weight(conf=conf, where=where, epoch=epoch, epochs=epochs, **kw)

        raise ValueError('{} is not a known placeholder for sample'.format(where))

    def calculate_conf(self, epoch=0, epochs=0):

        if self.in_setup_thr_on_val:
            return epoch <= self.switch_phase
        return epoch in (0, self.switch_phase, epochs)

    def init_epoch(self, net, data, conf, pred, epoch=0, epochs=0):

        super().init_epoch(net, data, conf, pred, epoch=epoch, epochs=epochs)

        if epoch in (0, self.switch_phase):
            self.reload_network(net)

        self.phase = 'liquid'
        if epoch < self.switch_phase:
            self.phase = 'gas'

        if not epoch and self.ft_checkpoint:
            try:
                net.load_state_dict(torch.load(self.ft_checkpoint))
                self.ft_checkpoint_loaded = True
                self.recorder.event('load_ft_state', 'done')
            except FileNotFoundError:
                self.recorder.event('load_ft_state', no='{} does not exist'.format(self.ft_checkpoint.name))

        if epoch == self.epochs and self.ft_checkpoint:
            torch.save(net.state_dict(), self.ft_checkpoint)

        if self.ft_checkpoint_loaded and self.phase == 'gas':
            self.phase = 'solid'
            return

        if self.phase == 'liquid' and self.in_setup_thr_on_val:
            self.phase = 'solid'
            return

        if not self.max_iterations:
            return

        padded_mix_size = sum(len(self.pad_buffers[_]) for _ in self.pad_buffers)

        if self.phase == 'liquid':
            padded_mix_size += int((conf < self.pad_thresholds['self']).sum())
        else:
            padded_mix_size += len(conf)

        it_per_epoch = padded_mix_size / self.batch_size
        epochs_per_phase = self.iterations_per_phase / it_per_epoch

        if self.phase == 'gas' and epoch >= epochs_per_phase:
            self.phase = 'solid'

        if self.phase == 'liquid' and (epoch - self.switch_phase) >= epochs_per_phase:
            self.phase = 'solid'

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0, pred=None):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        bias = net.get_fc_layer().bias

        # weight of shape CxK (eg C=10, L=64)
        weight = net.get_fc_layer().weight

        logits, features = net(data, return_feature=True)

        if pred is None:
            _, pred = torch.max(logits, dim=1)

        logits_y = logits.gather(-1, pred.unsqueeze(-1)).squeeze(-1)

        b_y = bias.index_select(0, pred)  # of shape M
        m_y_2 = weight.index_select(0, pred).square().sum(-1)  # of shape M (MxL summed on last dim)

        conf = logits_y - b_y - 0.5 * m_y_2

        return pred, conf
