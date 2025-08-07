from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class NeCOPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False

    @torch.no_grad
    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        net.eval()
        self.w, self.b = net.get_fc()
        print('Extracting id training feature')
        feature_id_train = []
        for batch in tqdm(id_loader_dict['train'],
                          desc='Setup: ',
                          position=0,
                          leave=True):
            data = batch['data'].cuda()
            data = data.float()
            _, feature = net(data, return_feature=True)
            feature_id_train.append(feature.cpu().numpy())

        feature_id_train = np.concatenate(feature_id_train, axis=0)
        logit_id_train = feature_id_train @ self.w.T + self.b

        self.ss = StandardScaler()
        complete_vectors_train = self.ss.fit_transform(feature_id_train)

        self.pca = PCA(feature_id_train.shape[1])
        _ = self.pca.fit_transform(complete_vectors_train)

        self.setup_flag = True

    @torch.no_grad
    def postprocess(self, net: nn.Module, data: Any):
        _, feature = net.forward(data, return_feature=True)
        feature = feature.cpu()
        logit = feature @ self.w.T + self.b
        _, pred = torch.max(logit, dim=1)

        complete_vectors = self.ss.transform(feature)
        cls_test_reduced_all = self.pca.transform(complete_vectors)

        score_ood = -vlogit_ood + energy_ood
        return pred, torch.from_numpy(score_ood)

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
