'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from typing import Union, List, cast

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2],
                                                            (0, 0, 0, 0, planes//4, planes//4),
                                                            "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                        kernel_size=1, stride=stride, bias=False),
                                              nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, momentum=0.03), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        gain = torch.nn.init.calculate_gain('relu')
        logits = torch.nn.Linear(64, num_classes)
        torch.nn.init.kaiming_normal_(logits.weight, gain)
        torch.nn.init.constant_(logits.bias, 0)
        self.linear = logits

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    # def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out_fea = out.view(out.size(0), -1)
        # out = self.linear(out_fea)
        # return out_fea, out

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature4 = F.avg_pool2d(feature4, feature4.size()[3])
        feature = feature4.view(feature4.size(0), -1)
        logits_cls = self.linear(feature)

        feature_list = [feature1, feature2, feature3, feature4]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def get_fc(self):
        linear = self.linear
        return linear.weight.cpu().detach().numpy(), linear.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.linear


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(
        list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":

    import time
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--net', default='resnet20')
    parser.add_argument('--batch', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('-N', type=int, default=1000)
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    def time_backprop(*batch_sizes, net_name='resnet20', Nt=1000, device='cuda'):

        device = device if torch.cuda.is_available() else 'cpu'
        net = eval(net_name)().to(device)

        ce = torch.nn.CrossEntropyLoss()

        net.train()

        print(net)

        for batch_size in batch_sizes:
            print('batch_size={}'.format(batch_size), end='', flush=True)
            x = torch.randn(batch_size, 3, 32, 32, device=device)
            y = torch.randint(10, (batch_size,), device=device)

            t0 = time.time()
            for i in range(Nt):

                if not (i % (Nt//10)):
                    print('.', end='', flush=True)
                out = net(x)
                loss = ce(out, y)
                loss.backward()

            t_unit = (time.time() - t0) / Nt

            print('\r{:4}: {:.3f}ms/batch = {:.3f}us/i'.format(batch_size, t_unit * 1e3,
                                                               t_unit / batch_size * 1e6))

    time_backprop(*args.batch, net_name=args.net, Nt=args.N, device=args.device)
