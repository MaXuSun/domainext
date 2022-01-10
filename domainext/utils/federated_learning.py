import copy
import torch
from torch import nn

def fedavg(w,weight=None):
    n = len(w)
    if weight is None:
        weight = [1/n]*n
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(n):
            if i == 0:
                w_avg[k] = weight[i]*w[i][k]
            else:
                w_avg[k] += (weight[i]*w[i][k])
    return w_avg
