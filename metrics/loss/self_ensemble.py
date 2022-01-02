"""
@misc{dalib,
  author = {Junguang Jiang, Baixu Chen, Bo Fu, Mingsheng Long},
  title = {Transfer-Learning-library},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thuml/Transfer-Learning-Library}},
}
"""
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'L2ConsistencyLoss',
    'ClassBalanceLoss'
]

class ConsistencyLoss(nn.Module):
    r"""
    Consistency loss between output of student model and output of teacher model.
    Given distance measure :math:`D`, student model's output :math:`y`, teacher
    model's output :math:`y_{teacher}`, binary mask :math:`mask`, consistency loss is

    .. math::
        D(y, y_{teacher}) * mask

    Args:
        distance_measure (callable): Distance measure function.
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    Inputs:
        - y: predictions from student model
        - y_teacher: predictions from teacher model
        - mask: binary mask

    Shape:
        - y, y_teacher: :math:`(N, C)` where C means the number of classes.
        - mask: :math:`(N, )` where N means mini-batch size.
    """

    def __init__(self, distance_measure: Callable, reduction: Optional[str] = 'mean'):
        super(ConsistencyLoss, self).__init__()
        self.distance_measure = distance_measure
        self.reduction = reduction

    def forward(self, y: torch.Tensor, y_teacher: torch.Tensor, mask: torch.Tensor):
        cons_loss = self.distance_measure(y, y_teacher)
        cons_loss = cons_loss * mask
        if self.reduction == 'mean':
            return cons_loss.mean()
        else:
            return cons_loss


class L2ConsistencyLoss(ConsistencyLoss):
    r"""
    L2 consistency loss. Given student model's output :math:`y`, teacher model's output :math:`y_{teacher}`
    and binary mask :math:`mask`, L2 consistency loss is

    .. math::
        \text{MSELoss}(y, y_{teacher}) * mask

    """

    def __init__(self, reduction: Optional[str] = 'mean'):
        def l2_distance(y: torch.Tensor, y_teacher: torch.Tensor):
            return ((y - y_teacher) ** 2).sum(dim=1)

        super(L2ConsistencyLoss, self).__init__(l2_distance, reduction)


class ClassBalanceLoss(nn.Module):
    r"""
    Class balance loss that penalises the network for making predictions that exhibit large class imbalance.
    Given predictions :math:`y` with dimension :math:`(N, C)`, we first calculate mean across mini-batch dimension,
    resulting in mini-batch mean per-class probability :math:`y_{mean}` with dimension :math:`(C, )`

    .. math::
        y_{mean}^j = \frac{1}{N} \sum_{i=1}^N y_i^j

    Then we calculate binary cross entropy loss between :math:`y_{mean}` and uniform probability vector :math:`u` with
    the same dimension where :math:`u^j` = :math:`\frac{1}{C}`

    .. math::
        loss = \text{BCELoss}(y_{mean}, u)

    Args:
        num_classes (int): Number of classes

    Inputs:
        - y (tensor): predictions from classifier

    Shape:
        - y: :math:`(N, C)` where C means the number of classes.
    """

    def __init__(self, num_classes):
        super(ClassBalanceLoss, self).__init__()
        self.uniform_distribution = torch.ones(num_classes) / num_classes

    def forward(self, y: torch.Tensor):
        return F.binary_cross_entropy(y.mean(dim=0), self.uniform_distribution.to(y.device))

