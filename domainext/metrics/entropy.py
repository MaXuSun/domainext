
import torch.nn.functional as F
import math
import torch

def entropy(predictions: torch.Tensor, reduction='none', epsilon=1e-5) -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:

    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c

    where C is number of classes.

    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output. Default: ``'mean'``

    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

def robust_entropy(y, ita=1.5, num_classes=19, reduction='mean'):
    """ Robust entropy proposed in `FDA: Fourier Domain Adaptation for Semantic Segmentation (CVPR 2020) <https://arxiv.org/abs/2004.05498>`_

    Args:
        y (tensor): logits output of segmentation model in shape of :math:`(N, C, H, W)`
        ita (float, optional): parameters for robust entropy. Default: 1.5
        num_classes (int, optional): number of classes. Default: 19
        reduction (string, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Returns:
        Scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    """
    P = F.softmax(y, dim=1)
    logP = F.log_softmax(y, dim=1)
    PlogP = P * logP
    ent = -1.0 * PlogP.sum(dim=1)
    ent = ent / math.log(num_classes)

    # compute robust entropy
    ent = ent ** 2.0 + 1e-8
    ent = ent ** ita

    if reduction == 'mean':
        return ent.mean()
    else:
        return ent