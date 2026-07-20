import csv
import os
from typing import Dict


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .ood_evaluator import OODEvaluator
from .metrics import compute_all_metrics

import pandas as pd


class TTAOODEvaluator(OODEvaluator):
    def __init__(self, config: Config):
        """OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(TTAOODEvaluator, self).__init__(config)
        self.tta_epochs = config.postprocessor.ft.get('epochs', 0)

        self.ood_period = config.pipeline.get('ood_period', 0)
        print('ood_period set to {}'.format(self.ood_period))

    def eval_ood(self,
                 net: nn.Module,
                 id_ood_aux_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor,
                 fsood: bool = False):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()

        if self.config.postprocessor.APS_mode:
            raise NotImplementedError

        splits = ('mixture',) if self.ood_period else ('nearood', 'farood')

        for ood_split in [_ for _ in splits if _ in id_ood_aux_data_loaders]:
            # load nearood data and compute ood metrics
            print(u'\u2500' * 70, flush=True)
            self._eval_ood(net,
                           id_ood_aux_data_loaders,
                           postprocessor,
                           ood_split=ood_split)

    def _eval_ood(self,
                  net: nn.Module,
                  id_ood_aux_data_loaders: Dict[str, Dict[str, DataLoader]],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood'):
        print(f'Processing {ood_split}...', flush=True)
        # set random seed

        tta_epochs = 0 if self.tta_epochs is None else self.tta_epochs

        df = self._init_df(id_ood_aux_data_loaders[ood_split], tta_epochs)

        torch.manual_seed(self.config.pipeline.seed)
        np.random.seed(self.config.pipeline.seed)

        for dataset_name, id_ood_dl in id_ood_aux_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)

            pred, conf, label = postprocessor.inference(net, id_ood_dl,
                                                        epochs=tta_epochs)

            for epoch in pred:
                if self.config.recorder.save_scores:
                    self._save_scores(pred[epoch], conf[epoch], label[epoch], dataset_name,
                                      epoch=epoch, epochs=self.tta_epochs)

            print(f'Computing metrics on {dataset_name} dataset...')
            self._update_df(df, id_ood_aux_data_loaders[ood_split], dataset_name, pred, conf, label)
            for epoch in pred:
                self._print_metrics(list(df.loc[(epoch, dataset_name)]),
                                    dataset_name, epoch, max(pred))

        if self.config.recorder.save_csv:
            for (epoch, dset) in df.index:
                self._save_csv(list(df.loc[(epoch, dset)]), dset,
                               epoch=epoch, epochs=self.tta_epochs)

    def _init_df(self, ood_data_loader, epochs):
        # Dataframe of results
        # results = [fpr, auroc, aupr_in, aupr_out, accuracy]
        columns = ['FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC']
        index = []
        for epoch in range(0, epochs + 1):
            for dataset_name, id_ood_dl in ood_data_loader.items():
                for sub_ood_idx, sub_ood in id_ood_dl.dataset.sub_ood.items():
                    index.append((epoch, sub_ood))
                if len(id_ood_dl.dataset.sub_ood) > 1:
                    index.append((epoch, dataset_name))

        index = pd.MultiIndex.from_tuples(index, names=('epoch', 'dataset'))
        return pd.DataFrame(index=index, columns=columns)

    def _update_df(self, df, ood_data_loader, dataset_name, pred, conf, label):

        id_ood_dl = ood_data_loader[dataset_name]
        for epoch in pred:

            ood_metrics = compute_all_metrics(conf[epoch], label[epoch], pred[epoch])
            df.loc[(epoch, dataset_name)] = ood_metrics

            for sub_ood_idx, sub_ood in id_ood_dl.dataset.sub_ood.items():
                kept_idx = (label[epoch] >= 0) | (label[epoch] == sub_ood_idx)
                ood_metrics = compute_all_metrics(conf[epoch][kept_idx],
                                                  label[epoch][kept_idx],
                                                  pred[epoch][kept_idx])

                df.loc[(epoch, sub_ood)] = ood_metrics

    def _save_csv(self, metrics, dataset_name, epoch=0, epochs=None):
        [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics
        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
            'ACC': '{:.2f}'.format(100 * accuracy)
        }

        fieldnames = list(write_content.keys())

        if epochs:
            fieldnames.insert(0, 'epoch')
            write_content['epoch'] = epoch

        csv_path = os.path.join(self.config.output_dir, 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _print_metrics(self, metrics, dataset_name, epoch=0, epochs=None):
        [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics
        # print ood metric results
        print('epoch {}/{}'.format(dataset_name, epoch, epochs))
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
            flush=True)
        print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
        print(u'\u2500' * 70, flush=True)

    def _save_scores(self, pred, conf, gt, save_name, epoch=0, epochs=None):
        save_dir = os.path.join(self.config.output_dir, 'scores')
        if epochs:
            save_dir += '-{:03d}'.format(epoch)
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 fsood: bool = False,
                 csid_data_loaders: DataLoader = None):
        """Returns the accuracy score of the labels and predictions.

        :return: float
        """
        if type(net) is dict:
            net['backbone'].eval()
        else:
            net.eval()
        id_pred, id_conf, id_gt = postprocessor.inference(net, data_loader)

        print('***', id_pred)
        self.id_pred = id_pred[0]
        self.id_conf = id_conf[0]
        self.id_gt = id_gt[0]

        if fsood:
            raise NotImplementedError
            assert csid_data_loaders is not None
            for dataset_name, csid_dl in csid_data_loaders.items():
                csid_pred, csid_conf, csid_gt = postprocessor.inference(
                    net, csid_dl)
                self.id_pred = np.concatenate([self.id_pred, csid_pred])
                self.id_conf = np.concatenate([self.id_conf, csid_conf])
                self.id_gt = np.concatenate([self.id_gt, csid_gt])

        metrics = {}
        metrics['acc'] = sum(self.id_pred == self.id_gt) / len(self.id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics
