import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self, n, in_channels, out_channels):

        super().__init__()
        ins = [out_channels] * n
        ins[0] = in_channels
        outs = [out_channels] * n

        self.layers = nn.ModuleList([nn.Conv2d(in_channels=i, out_channels=o, kernel_size=3, padding=1)
                                     for i, o in zip(ins, outs)])

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        for l in self.layers:

            x = l(x)
            x = F.relu(x)

        return self.pool(x)


class VGG19(nn.Module):

    def __init__(self, num_classes=10, input_shape=(32, 32), latent_dim=512):

        super().__init__()

        block_params = [(2, 3, 64), (2, 64, 128), (4, 128, 256), (4, 256, 512), (4, 512, 512)]
        self.blocks = nn.ModuleList([Block(n, i, o) for n, i, o in block_params])

        out_shape = int((input_shape[0] / 2**len(block_params)) ** 2 * block_params[-1][-1])

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(out_shape, latent_dim)
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x, layer_index=None,
                return_feature_list=False, return_feature=False,
                threshold=None,
                ):

        feature_list = []
        if layer_index is None:
            layer_index = len(self.blocks) + 1

        for i, b in enumerate(self.blocks):

            print(i, layer_index)
            x = b(x)
            if return_feature_list:
                feature_list.append(x)
            if i + 1 == layer_index:
                return x

        if return_feature_list:
            return feature_list

        x = self.flatten(x)

        z = self.fc1(x)
        z = F.relu(z)

        if threshold is not None:
            z = z.clip(max=threshold)

        if return_feature:
            return z

        return self.fc(z)

    def intermediate_forward(self, x, layer_index):
        return self.forward(x, layer_index=layer_index)

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc
