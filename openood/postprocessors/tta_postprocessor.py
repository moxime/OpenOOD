import os
from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm

from .base_postprocessor import BasePostprocessor


def _unfold_(obj, prefix='', prefix_inc='  '):

    if isinstance(obj, (list, tuple)):
        for _ in obj:
            _unfold_(obj, prefix=prefix+prefix_inc, prefix_inc=prefix_inc)

        return

    if isinstance(obj, dict):

        for _ in obj:
            print(prefix + _)
            _unfold_(obj[_], prefix=prefix+prefix_inc, prefix_inc=prefix_inc)

        return

    str_ = type(obj)

    if isinstance(obj, torch.utils.data.dataloader.DataLoader):
        str_ = '{N}= {n}x{m} s: {s} from {f}'.format(n=len(obj), m=obj.batch_size,
                                                     N=len(obj) * obj.batch_size,
                                                     s=type(obj.sampler).__name__[0],
                                                     f=obj.dataset)

    print(prefix + str_)


class TTAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        self.setup_flag = True

    def postprocess(self, net: nn.Module, data: Any):

        with torch.no_grad():
            output = net(data)
            score = torch.softmax(output, dim=1)
            conf, pred = torch.max(score, dim=1)

        """if there is some finentuning, do it here (will be done tta_epochs +1)
            
        - the first result is for 0 finetuning
            
        - the last result yielded will be after tta_epochs finetuning
            
        """

        minibatch_loader = DataLoader(data, shuffle=True,
                                      batch_size=self.config.postprocessor.tta_batch_size)
        self.finetune(net, minibatch_loader)

        return pred, conf

    def finetune(self, net, data):
        pass

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  epochs=0,
                  progress: bool = True):

        lists = {_: {} for _ in ('pred', 'conf', 'label')}

        try:
            reset_net_at_epoch = self.config.postprocessor.reset_network
        except AttributeError:
            reset_net_at_epoch = True
        if reset_net_at_epoch:
            reset_net_at_epoch = os.path.join(self.config.output_dir, 'tmp.ckpt')
            net.save(reset_net_at_epoch)
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            if reset_net_at_epoch:
                net.load(reset_net_at_epoch)
            for epoch in range(epochs+1):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                pred, conf = self.postprocess(net, data)

                for key, tensor in zip(('pred', 'conf', 'label'), (pred, conf, label)):
                    lists[key].setdefault(epoch, []).append(tensor.cpu())

            # convert values into numpy array
        for _ in lists:
            for epoch in lists[_]:
                lists[_][epoch] = torch.cat(lists[_][epoch]).numpy().astype(
                    float if _ == 'conf' else int)

        return tuple(lists[_] for _ in ('pred', 'conf', 'label'))
