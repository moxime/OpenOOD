import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):

    def __init__(self, size, in_channels, out_channels):

        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(size):

            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(out_channels))
            self.layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        for l in self.layers:

            x = l(x)

        return self.pool(x)


class PriorMean(nn.Module):

    def __init__(self, num_classes=10, latent_space=1024):

        super().__init__()
        self.mean = nn.Parameter(torch.Tensor(num_classes, latent_space))

    def get_fc(self):

        fc = nn.Linear(self.mean.shape[1], self.mean.shape[0])
        fc.weight = nn.Parameter(-self.mean)
        fc.bias = nn.Parameter(0.5 * self.mean.norm(dim=1).pow(2))

        return fc

    def forward(self, z, debug=False):
        """
        mu (mean) of shape CxK
        z of shape NxK
        outputs logit of shape NxC
        with logit[i, k] = 0.5 ||z[i,:]||² + 0.5 ||mu[k, :]||² - <z[i,:], mu[k,:]>
        """
        z_norm = 0.5 * z.norm(dim=1).pow(2).unsqueeze(1)
        bias = 0.5 * self.mean.norm(dim=1).pow(2).unsqueeze(0)

        val_ = torch.matmul(z, self.mean.T)
        out = val_ - z_norm - bias

        if debug:
            def isinf(t):
                return 'inf' if t.isinf().any() else 'ok'
            print('z', *z.shape, '| z_norm', *z_norm.shape, '| b', *bias.shape,
                  '| val_', *val_.shape, '| out', *out.shape)
            print('z', isinf(z), '| z_norm', isinf(z_norm), '| b', isinf(bias),
                  '| val_', isinf(val_), '| out', isinf(out))

        return out


class VGG19CVAE(nn.Module):

    def __init__(self, num_classes=10, input_shape=(32, 32),
                 encoder=[],
                 latent_dim=1024, relu_after_latent=False):

        super().__init__()

        block_params = [(2, 3, 64), (2, 64, 128), (4, 128, 256), (4, 256, 512), (4, 512, 512)]
        self.features = nn.ModuleList([VGGBlock(n, i, o) for n, i, o in block_params])
        self.features.append(nn.AvgPool2d(1))

        out_shape = int((input_shape[0] / 2**len(block_params)) ** 2 * block_params[-1][-1])

        self.flatten = nn.Flatten()

        self.encoder = nn.ModuleDict({'dense_projs': nn.Sequential()})

        for dim in encoder:
            self.encoder['dense_projs'].append(nn.Linear(out_shape, dim))
            self.encoder['dense_projs'].append(nn.ReLU(inplace=True))
            out_shape = dim

        self.encoder['dense_mean'] = nn.Linear(out_shape, latent_dim)

        self.encoder['prior'] = PriorMean()

    def _transfer_state_dict(self, other_state_dict):

        categories = ('encoder', 'features')

        other_state_dict_dict = {k: {_: other_state_dict[_] for _ in other_state_dict if _.startswith(k + '.')}
                                 for k in categories}

        self_state_dict = self.state_dict()

        self_keys = {cat: [_ for _ in self_state_dict if _.startswith(cat+'.')] for cat in categories}

        # dict of {self_key: other_key}
        key_map = {}

        category = 'features'
        for self_k, other_k in zip(self_keys[category], other_state_dict_dict[category]):
            key_map[self_k] = other_k

        for category in ('encoder',):

            for k in self_keys[category]:
                if k in other_state_dict:
                    key_map[k] = k

        for self_k, other_k in key_map.items():
            other_shape = other_state_dict[other_k].shape
            self_shape = self_state_dict[self_k].shape
            if other_shape != self_shape:
                raise ValueError('{} shape ({}) is != from {} shape ({})'.format(other_k, other_shape,
                                                                                 self_k, self_shape))

        missing_keys = {_ for _ in self_state_dict if _ not in key_map}

        if missing_keys:
            raise ValueError('Missing keys: {}'.format(' '.join(missing_keys)))

        unused_keys = {k: {_ for _ in other_state_dict_dict[k] if _
                           not in key_map.values()} for k in self_keys}

        print(unused_keys)
        return {k: other_state_dict[key_map[k]] for k in key_map}

    def load_state_dict(self, state_dict, **kw):

        new_state = self._transfer_state_dict(state_dict)
        return super().load_state_dict(new_state, **kw)

    def forward(self, x, layer_index=None,
                return_feature_list=False, return_feature=False,
                threshold=None):

        feature_list = []
        if layer_index is None:
            layer_index = len(self.features) + 1

        for i, b in enumerate(self.features):

            x = b(x)
            if return_feature_list:
                feature_list.append(x)
            if i + 1 == layer_index:
                return x

        x = self.flatten(x)

        z = self.encoder['dense_projs'](x)
        z = self.encoder['dense_mean'](z)

        if threshold is not None:
            z = z.clip(max=threshold)

        logits = self.encoder.prior(z)

        if return_feature_list:
            return logits, feature_list

        if return_feature:
            return logits, z

        return logits

    def intermediate_forward(self, x, layer_index):
        return self.forward(x, layer_index=layer_index)

    def forward_threshold(self, x, threshold):

        return self.forward(x, threshold=threshold)

    def get_fc(self):
        fc = self.encoder.prior.get_fc()
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.encoder.prior.get_fc()


if __name__ == '__main__':

    import torch
    import numpy as np
    m = VGG19CVAE(latent_dim=1024)
    state_dict = torch.load('state.pth', map_location='cpu')

    state = m.state_dict()
    k = list(state)[0]
    param = state[k].clone()

    m.load_state_dict(state_dict)

    new_param = m.state_dict()[k].clone()

    print((param - new_param).norm())

    x = torch.randn(100, 3, 32, 32)

    logits = m(x)

    print(logits.shape)
