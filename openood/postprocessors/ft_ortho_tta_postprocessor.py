from typing import Any

import torch
import torch.nn as nn

from .ft_dist_tta_postprocessor import DistTTAPostprocessor


class OrthoTTAPostprocessor(DistTTAPostprocessor):

    def adaptation_loss(self, logits, features, net):

        bias = net.get_fc_layer().bias

        wz = logits - bias

        return wz.abs().mean(-1)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, epoch=0, pred=None):
        """ postprocess is done for each "chunk" of data at each epoch
        """

        bias = net.get_fc_layer().bias
        logits, features = net(data, return_feature=True)
        softmax_probs = torch.softmax(logits, dim=1)

        if pred is None:
            _, pred = torch.max(logits, dim=1)

        ortho_1 = (logits - bias).abs().mean(-1)

        prob_y = torch.gather(softmax_probs, -1, pred.unsqueeze(-1)).squeeze(-1)

        log_prob_y = prob_y.log()
        conf = log_prob_y + ortho_1

        self.recorder.event('ortho_comp', len=len(conf), ortho='{:.2g}'.format(ortho_1.mean()),
                            log_prob='{:.2g}'.format(log_prob_y.mean()))

        return pred, conf
