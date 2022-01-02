
from ..common.torchtools import set_requires_grad
from typing import Optional
import copy
import torch

class EmaTeacher(object):
    r"""
    Exponential moving average model used in `Self-ensembling for Visual Domain Adaptation (ICLR 2018) <https://arxiv.org/abs/1706.05208>`_

    We denote :math:`\theta_t'` as the parameters of teacher model at training step t, :math:`\theta_t` as the
    parameters of student model at training step t, :math:`\alpha` as decay rate. Then we update teacher model in an
    exponential moving average manner as follows

    .. math::
        \theta_t'=\alpha \theta_{t-1}' + (1-\alpha)\theta_t

    Args:
        model (torch.nn.Module): student model
        alpha (float): decay rate for EMA.

    Inputs:
        x (tensor): input data fed to teacher model

    Examples::

        >>> classifier = ImageClassifier(backbone, num_classes=31, bottleneck_dim=256).to(device)
        >>> # initialize teacher model
        >>> teacher = EmaTeacher(classifier, 0.9)
        >>> num_iterations = 1000
        >>> for _ in range(num_iterations):
        >>>     # x denotes input of one mini-batch
        >>>     # you can get teacher model's output by teacher(x)
        >>>     y_teacher = teacher(x)
        >>>     # when you want to update teacher, you should call teacher.update()
        >>>     teacher.update()
    """

    def __init__(self, model, alpha):
        self.model = model
        self.alpha = alpha
        self.teacher = copy.deepcopy(model)
        set_requires_grad(self.teacher, False)

    def set_alpha(self, alpha: float):
        assert alpha >= 0
        self.alpha = alpha

    def update(self):
        for teacher_param, param in zip(self.teacher.parameters(), self.model.parameters()):
            teacher_param.data = self.alpha * teacher_param + (1 - self.alpha) * param

    def __call__(self, x: torch.Tensor):
        return self.teacher(x)

    def train(self, mode: Optional[bool] = True):
        self.teacher.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.teacher.state_dict()

    def load_state_dict(self, state_dict):
        self.teacher.load_state_dict(state_dict)

    @property
    def module(self):
        return self.teacher.module
