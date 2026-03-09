import torch.nn as nn
import torch


def ortho_weights(weights, features):

    return torch.matmul(features, weights.T).abs().mean(dim=-1)


def uniform_ce(logits, **kw):
    """

        returns CE(unif, p(y|x)):

            CE = -\frac 1 {num_class} \sum_y log p(y|x) 
               = -output.softmax(-1).log()


        that can be calculated as:
                -output.mean(-1) + output.logsumexp(-1)

        """
    return -logits.mean(-1) + logits.logsumexp(-1)


class MarginCrossEntropy(nn.CrossEntropyLoss):

    def __init__(self, *a, margin=0, **kw):

        super().__init__(*a, **kw)
        self.margin = margin

    def forward(self, input, target):

        if self.margin:
            margin = 1. + torch.zeros_like(input).scatter_(1, target.unsqueeze(1), self.margin)
        else:
            margin = 1.

        return super().forward(input * margin, target)


if __name__ == '__main__':

    import argparse

    N = 100
    C = 10
    K = 16

    logits = torch.randn(N, C)
    labels = torch.randint(C, (N,))

    parser = argparse.ArgumentParser()
    parser.add_argument('margin', type=float)
    parser.add_argument('-a', type=float, default=20)

    args = parser.parse_args()
    logits += torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), args.a)

    loss = MarginCrossEntropy(margin=args.margin)

    msp = logits.softmax(-1).max(-1)[0]

    print(' -- '.join(map('{:.1%}'.format, msp[:10])))
    print('L = {:.1g}'.format(loss(logits, labels)))
