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


class VGG19(nn.Module):

    def __init__(self, num_classes=10, input_shape=(32, 32), latent_dim=1024, relu_after_fc=False):

        super().__init__()

        block_params = [(2, 3, 64), (2, 64, 128), (4, 128, 256), (4, 256, 512), (4, 512, 512)]
        self.blocks = nn.ModuleList([VGGBlock(n, i, o) for n, i, o in block_params])
        self.blocks.append(nn.AvgPool2d(1))

        out_shape = int((input_shape[0] / 2**len(block_params)) ** 2 * block_params[-1][-1])

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(out_shape, latent_dim)
        self.fc1_out = nn.ReLU(inplace=True) if relu_after_fc else nn.Identity(inplace=True)
        self.classifer = nn.ModuleList([nn.Linear(latent_dim, num_classes)])

    def forward(self, x, layer_index=None,
                return_feature_list=False, return_feature=False,
                threshold=None,
                ):

        feature_list = []
        if layer_index is None:
            layer_index = len(self.blocks) + 1

        for i, b in enumerate(self.blocks):

            x = b(x)
            if return_feature_list:
                feature_list.append(x)
            if i + 1 == layer_index:
                return x

        x = self.flatten(x)

        z = self.fc1(x)
        z = self.fc1_out(z)

        if threshold is not None:
            z = z.clip(max=threshold)

        for l in self.classifer:
            z = l(z)
        logits = z

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
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc


if __name__ == '__main__':

    m = VGG19(latent_dim=1024)

    for _ in m.state_dict():
        print(_)
