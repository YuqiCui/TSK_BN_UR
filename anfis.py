import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init


class GaussianMF(nn.Module):
    def __init__(self, in_dims, n_rules, init_mu=None, ampli=0):
        super(GaussianMF, self).__init__()
        self.n_rules = n_rules
        self.in_dims = in_dims
        self.ampli = ampli
        self.build()
        if init_mu is not None:
            self.mu.data = torch.from_numpy(init_mu).float()

    def build(self):
        self.mu = nn.Parameter(torch.FloatTensor(self.in_dims, self.n_rules), requires_grad=True)
        self.sig = nn.Parameter(torch.FloatTensor(self.in_dims, self.n_rules), requires_grad=True)
        Init.constant_(self.mu, 0)
        Init.constant_(self.sig, 1)

    def forward(self, x):
        frs = torch.exp(torch.sum(-(x.unsqueeze(dim=2) - self.mu) ** 2 / (2 * self.sig**2), dim=1) + self.ampli)
        return frs


class ANFIS(nn.Module):
    def __init__(self, in_dims, out_dims, n_rules, init_mu=None, ampli=0):
        super(ANFIS, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.n_rules = n_rules
        self.eps = 1e-8  # avoid numeric underflow
        self.build(init_mu, ampli)

    def build(self, init_mu, ampli):
        self.MFs = GaussianMF(self.in_dims, self.n_rules, init_mu, ampli)
        self.cons = []
        for i in range(self.n_rules):
            self.add_module('cons'+str(i), nn.Linear(self.in_dims, self.out_dims))
            self.cons.append(eval('self.cons{}'.format(i)))

    def defuzzy(self, frs):
        frs = frs / (torch.sum(frs, dim=1, keepdim=True) + self.eps)
        return frs

    def forward(self, x):
        frs = self.MFs(x)
        defuzzy_frs = self.defuzzy(frs)
        cons = torch.cat([cons(x).unsqueeze(dim=1) for cons in self.cons], dim=1)
        cons_out = torch.sum(torch.mul(defuzzy_frs.unsqueeze(dim=2), cons), dim=1, keepdim=False)
        return cons_out
