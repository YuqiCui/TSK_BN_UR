import torch as t
import torch.nn as nn
import torch.nn.init as Init
import skfuzzy
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class ClsTSK(nn.Module):
    def __init__(self, in_dim, n_rules, n_classes, init_centers, bn=False, init_Vs=None, ampli=0,):
        super(ClsTSK, self).__init__()
        self.in_dim = in_dim
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.init_centers = init_centers
        self.init_Vs = init_Vs
        self.ampli = ampli
        if bn:
            self.bn1 = nn.BatchNorm1d(num_features=self.in_dim)

        self.build_model()

    def rebuild_model(self, cuda=False):
        self.Cs.data = t.from_numpy(self.init_centers).float()
        if self.init_Vs is not None:
            self.Vs.data = t.from_numpy(self.init_Vs).float()
        else:
            Init.normal_(self.Vs, mean=1, std=0.2)
        Init.uniform_(self.Cons, -1, 1)
        Init.constant_(self.Bias, 0)

        if cuda:
            self.cuda()

    def build_model(self):
        self.eps = 1e-8

        self.Cons = t.FloatTensor(size=(self.n_rules, self.in_dim, self.n_classes))
        self.Bias = t.FloatTensor(size=(1, self.n_rules, self.n_classes))
        self.Cs = t.FloatTensor(size=(self.in_dim, self.n_rules))
        self.Vs = t.FloatTensor(size=self.Cs.size())

        self.Cons = nn.Parameter(self.Cons, requires_grad=True)
        self.Bias = nn.Parameter(self.Bias, requires_grad=True)
        self.Cs = nn.Parameter(self.Cs, requires_grad=True)
        self.Vs = nn.Parameter(self.Vs, requires_grad=True)

        self.Cs.data = t.from_numpy(self.init_centers).float()
        if self.init_Vs is not None:
            self.Vs.data = t.from_numpy(self.init_Vs).float()
        else:
            Init.normal_(self.Vs, mean=1, std=0.2)
        Init.uniform_(self.Cons, -1, 1)
        Init.constant_(self.Bias, 0)

    def forward(self, x, with_frs=False):
        if hasattr(x, 'rbn'):
            x = self.rbn(x)
        frs = t.exp(
            t.sum(
                -(x.unsqueeze(dim=2) - self.Cs) ** 2 / (2 * self.Vs ** 2), dim=1
            ) + self.ampli
        )
        frs = frs / (t.sum(frs, dim=1, keepdim=True) + self.eps)
        if hasattr(self, 'bn1'):
            x = self.bn1(x)
        x_rep = x.unsqueeze(dim=1).expand([x.size(0), self.n_rules, x.size(1)])
        cons = t.einsum('ijk,jkl->ijl', [x_rep, self.Cons])
        cons = cons + self.Bias
        cons = t.mul(cons, frs.unsqueeze(2))
        if with_frs:
            return t.sum(cons, dim=1, keepdim=False), frs
        return t.sum(cons, dim=1, keepdim=False)

    def l2_loss(self):
        return t.sum(self.Cons ** 2)

    def ur_loss(self, frs):
        return ((t.mean(frs, dim=0) - 1/self.n_classes)**2).sum()


