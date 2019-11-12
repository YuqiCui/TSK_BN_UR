import torch as t
import torch.nn as nn
import torch.optim as Optim
import torch.utils.data as Data

from lib.optim import AdaBound


def get_optim_func(optim_type):
    support_optim = ['adabound', 'sgd', 'adam']
    if optim_type == 'adabound':
        optim_func = AdaBound
    elif optim_type == 'sgd':
        optim_func = Optim.SGD
    elif optim_type == 'adam':
        optim_func = Optim.Adam
    else:
        raise ValueError('Unsupport optim type: {}, only supporting: {}'.format(optim_type, support_optim))
    return optim_func


def get_loss_func(loss, regression, n_classes=None):
    if regression and loss == 'mse':
        return nn.MSELoss()
    elif not regression and loss == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif not regression and loss == 'mse':
        return ClassificationMSELoss(n_classes)


def data_loader(data, longlabel=False, batch_size=512, shuffle=True):
    t_data = []
    for n in range(len(data)):
        t_data.append(t.from_numpy(data[n]).float())
    if longlabel:
        t_data[-1] = t_data[-1].long()

    tenDataset = Data.TensorDataset(*t_data)
    return Data.DataLoader(
        dataset=tenDataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


class ClassificationMSELoss(nn.Module):
    def __init__(self, n_classes):
        super(ClassificationMSELoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, outs, labels):
        one_hot = t.zeros(outs.size(0), self.n_classes)
        if labels.is_cuda:
            one_hot = one_hot.cuda()
        one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1)
        return t.mean(t.sum((outs - one_hot) ** 2, dim=1), dim=0, keepdim=False)
