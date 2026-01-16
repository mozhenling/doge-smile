#coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.ERM import ERM

class ELMloss(ERM):
    """
    Ref.:
        [1] S. Kato and K. Hotta, “Enlarged Large Margin Loss for Imbalanced Classification,
           ” Jun. 15, 2023, arXiv: arXiv:2306.09132. doi: 10.48550/arXiv.2306.09132.

        [2] https://github.com/usagisukisuki/ELMloss/blob/main/loss.py
    """
    def __init__(self, config, train_examples, adapt_examples=None):
        super(ELMloss, self).__init__(config, train_examples, adapt_examples)
        self.config = config
        self.cls_num_list = [len([l for l in self.train_set["label"] if l == i]) for i in range(self.num_classes)]
        self.loss  = ELMLossBase(self.cls_num_list)



class ELMLossBase(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, lambd=1.0, weight=None, s=30):
        super(ELMLossBase, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.lambd = lambd

    def forward(self, x, target):
        ### true class ###
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)

        ### maximum other class ###
        x_ = x.clone()
        ones = torch.ones_like(x_, dtype=torch.uint8) * 1e+8 * -1
        x_ = torch.where(index, ones, x_)
        x_ = x_.argmax(dim=1)
        index2 = torch.zeros_like(x, dtype=torch.uint8)
        index2.scatter_(1, x_.data.view(-1, 1), 1)
        index_float2 = index2.type(torch.cuda.FloatTensor)

        ### settting large margin ###
        batch_m1 = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m1 = batch_m1.view((-1, 1))
        batch_m2 = torch.matmul(self.m_list[None, :], index_float2.transpose(0,1))
        batch_m2 = batch_m2.view((-1, 1))

        x_m = x - batch_m1 + batch_m2 * self.lambd
        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s*output, target, weight=self.weight)
