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
    return -logits.mean() + logits.logsumexp(-1).mean()
