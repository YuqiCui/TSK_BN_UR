from lib.torch_utils import get_loss_func, get_optim_func, data_loader
import torch as t
from time import time
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import StepLR


def eval_acc(model, loader, cuda=False):
    model.eval()
    num_correct = 0
    num_data = 0
    for s, (inputs, labels) in enumerate(loader):
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        out = model(inputs)
        pred = t.argmax(out, dim=1)
        num_correct += t.sum(pred == labels).item()
        num_data += labels.size(0)
    return num_correct / num_data


def eval_bca(model, loader, cuda=False):
    model.eval()
    outs, trues = [], []
    for s, (inputs, labels) in enumerate(loader):
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        out, frs = model(inputs, True)
        pred = t.argmax(out, dim=1)
        outs.append(pred)
        trues.append(labels)
    return balanced_accuracy_score(
        t.cat(trues, dim=0).detach().cpu().numpy(),
        t.cat(outs, dim=0).detach().cpu().numpy()
    )


def eval_frs(model, loader, cuda=False):
    model.eval()
    frss = []
    for s, (inputs, labels) in enumerate(loader):
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        out, frs = model(inputs, with_frs=True)
        frss.append(frs)
    frss = t.cat(frss, dim=0).detach().cpu().numpy()
    return frss


def eval_raw_frs(model, loader, cuda=False):
    model.eval()
    frss = []
    for s, (inputs, labels) in enumerate(loader):
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        frs = model.raw_firing_level(inputs)
        frss.append(frs)
    frss = t.cat(frss, dim=0).detach().cpu().numpy()
    return frss


class ClassModelTrain():
    def __init__(self, model, train_data, eval_data, test_data=None,
                 n_classes=None, optim_type='adabound', args=None, save_path='ckpt/tmp.pkl'):
        if args is None:
            raise ValueError('Args can\'t be None')
        self.optim_func = get_optim_func(optim_type)
        self.loss_func = get_loss_func(args.loss_type, regression=False, n_classes=n_classes)
        self.save_path = save_path
        self.args = args
        if args.gpu:
            model.cuda()
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data

    def train(self):
        optim = self.optim_func(self.model.parameters(), lr=self.args.lr)
        best_acc, count, best_test_acc = 0, 0, 0
        trainer = data_loader(self.train_data, True, batch_size=self.args.batch_size)
        valer = data_loader(self.eval_data, True, batch_size=self.args.batch_size, shuffle=False)
        if self.test_data is not None:
            tester = data_loader(self.test_data, True, batch_size=self.args.batch_size, shuffle=False)
        hist = {
            'val_acc': [],
            'test_acc': [],
        }

        for e in range(self.args.epochs):
            self.model.train()
            start_t = time()
            for s, (inputs, labels) in enumerate(trainer):
                if self.args.gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                out, frs = self.model(inputs, with_frs=True)
                loss = self.loss_func(out, labels) + \
                       self.args.weight_decay * self.model.l2_loss() + \
                       self.args.weight_frs * self.model.ur_loss(frs) + \
                       self.args.weight_div * self.model.cons_diversity()

                optim.zero_grad()
                loss.backward()
                optim.step()
            val_acc = eval_acc(self.model, valer, self.args.gpu)
            hist['val_acc'].append(val_acc)
            if self.test_data is not None:
                test_acc = eval_acc(self.model, tester, self.args.gpu)
                hist['test_acc'].append(test_acc)
            else:
                test_acc = 0
            end_t = time()

            print('\r[FLAG {:2d}][TRAIN {:4d}] Val ACC: {:.4f}, Test ACC: {:.4f}, Best Val ACC: {:.4f}, '
                  'Best Test ACC: {:.4f}, Time: {:.2f}s'.format(
                self.args.flag, e, val_acc, test_acc, best_acc, best_test_acc, end_t - start_t), end='')

            if val_acc > best_acc:
                best_acc = val_acc
                best_test_acc = test_acc
                count = 0
                t.save(self.model.state_dict(), self.save_path)
            else:
                count += 1
                if count > self.args.patience:
                    break
        print()
        self.model.load_state_dict(t.load(self.save_path))
        best_test_bca = eval_bca(self.model, tester, self.args.gpu)
        print('Test BCA: {:.4f}'.format(best_test_bca))
        return hist, best_test_acc, best_test_bca

    def tune_ur_param(self):
        self.model.rebuild_model(self.args.gpu)
        t_range = [0.1, 1, 10, 20, 50]
        hist = [1] * len(t_range)
        best_val_acc = np.zeros([len(t_range)])
        test_acc = np.zeros([len(t_range)])
        self.tmp_save_path = self.save_path
        for k, weight_frs in enumerate(t_range):
            self.save_path = self.tmp_save_path + '.tune_w{}'.format(weight_frs)
            self.args.weight_frs = weight_frs
            hist[k], test_acc[k] = self.train()
            best_val_acc[k] = np.max(hist[k]['val_acc'])
        best_index = np.argmax(best_val_acc)
        print('[FLAG {:2d}][TUNING] Best frs weight: {}'.format(self.args.flag, t_range[best_index]))
        os.rename(self.tmp_save_path + '.tune_w{}'.format(t_range[best_index]), self.tmp_save_path)
        hist[best_index]['weight_frs'] = t_range[best_index]
        return hist[best_index], test_acc[best_index]

    def tune_div_param(self):
        self.model.rebuild_model(self.args.gpu)
        t_range = [0.01, 0.1, 1, 10, 20]
        hist = [1] * len(t_range)
        best_val_acc = np.zeros([len(t_range)])
        test_acc = np.zeros([len(t_range)])
        self.tmp_save_path = self.save_path
        for k, weight_div in enumerate(t_range):
            self.save_path = self.tmp_save_path + '.tune_div{}'.format(weight_div)
            self.args.weight_div = weight_div
            hist[k], test_acc[k] = self.train()
            best_val_acc[k] = np.max(hist[k]['val_acc'])
        best_index = np.argmax(best_val_acc)
        print('[FLAG {:2d}][TUNING] Best div weight: {}'.format(self.args.flag, t_range[best_index]))
        os.rename(self.tmp_save_path + '.tune_div{}'.format(t_range[best_index]), self.tmp_save_path)
        hist[best_index]['weight_div'] = t_range[best_index]
        return hist[best_index], test_acc[best_index]

    def tune_ur_div_param(self):
        self.model.rebuild_model(self.args.gpu)
        ur_range, div_range = [0.1, 1, 10, 20, 50], [0.01, 0.1, 1, 10, 20]
        param_range = list(zip(ur_range, div_range))
        hist = [1] * len(param_range)
        best_val_acc, test_acc = np.zeros([len(param_range)]), np.zeros([len(param_range)])
        self.tmp_save_path = self.save_path
        for k, (ur_weight, div_weight) in enumerate(param_range):
            self.save_path = self.tmp_save_path + '_tune_div{}_ur{}'.format(div_weight, ur_weight)
            self.args.weight_div, self.args.weight_frs = div_weight, ur_weight
            hist[k], test_acc[k] = self.train()
            best_val_acc[k] = np.max(hist[k]['val_acc'])
        best_index = np.argmax(best_val_acc)
        print('[FLAG {:2d}][TUNING] Best div weight: {} Best ur weight: {}'.format(
            self.args.flag, param_range[best_index][1], param_range[best_index][0]))
        os.rename(self.tmp_save_path + '_tune_div{}_ur{}'.format(param_range[best_index][1], param_range[best_index][0]), self.tmp_save_path)
        hist[best_index]['weight_div'] = param_range[best_index][1]
        hist[best_index]['weight_frs'] = param_range[best_index][0]
        return hist[best_index], test_acc[best_index]
