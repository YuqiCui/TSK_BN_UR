from lib.torch_utils import get_loss_func, get_optim_func, data_loader
import torch as t
from time import time
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import itertools


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
        out = model(inputs)
        pred = t.argmax(out, dim=1)
        outs.append(pred)
        trues.append(labels)
    return balanced_accuracy_score(
        t.cat(trues, dim=0).detach().cpu().numpy(),
        t.cat(outs, dim=0).detach().cpu().numpy()
    )


class ClassModelTrain():
    def __init__(self, model, train_data, test_data=None,
                 n_classes=None, optim_type='adabound',
                 args=None, save_path='tmp.pkl'):
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
        self.test_data = test_data

    def split_train_val(self, val_size, random_state=None):
        x_train, y_train = self.train_data
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_size, random_state=random_state
        )
        return x_train, y_train, x_val, y_val

    def train(self):
        if self.args.weight_frs > 0:
            range_ur = [0.1,  1, 10, 20, 50]
            bcas = np.zeros([len(range_ur)])
            stops = np.zeros([len(range_ur)])
            for i in range(self.args.repeats):
                for k, w_ur in enumerate(range_ur):
                    bca, _, pos = self._train_(self.args.weight_decay, w_ur=w_ur)
                    bcas[k] += bca
                    stops[k] += pos
            bcas /= self.args.repeats
            stops /= self.args.repeats
            best_idx = int(np.argmax(bcas))
            print('\nbest ur param: {}, best stop pos: {}'.format(
                range_ur[best_idx], stops[best_idx]
            ))
            bca, acc = self.train_final_model(self.args.weight_decay, range_ur[best_idx], int(round(stops[best_idx])))
            return bca, acc
        stops = 0
        for i in range(self.args.repeats):
            _, _, pos = self._train_(self.args.weight_decay, w_ur=0)
            stops += pos
        stops /= self.args.repeats
        print('\nbest stop pos: {}'.format(stops))
        bca, acc = self.train_final_model(self.args.weight_decay, 0, int(round(stops)))
        return bca, acc

    def train_final_model(self, w_l2, w_ur, epochs):
        x_train, y_train = self.train_data
        self.model.rebuild_model(self.args.gpu)
        optim = self.optim_func(self.model.parameters(), lr=self.args.lr)
        tester = data_loader(self.test_data, True, batch_size=self.args.batch_size, shuffle=False)
        trainer = data_loader([x_train, y_train], True, batch_size=self.args.batch_size)
        for e in range(epochs):
            self.model.train()
            for s, (inputs, labels) in enumerate(trainer):
                if self.args.gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                out, frs = self.model(inputs, with_frs=True)
                loss = self.loss_func(out, labels) + \
                       w_l2 * self.model.l2_loss() + \
                       w_ur * self.model.ur_loss(frs)
                optim.zero_grad()
                loss.backward()
                optim.step()
        t.save(self.model.state_dict(), self.save_path)
        best_test_acc = eval_acc(self.model, tester, self.args.gpu)
        best_test_bca = eval_bca(self.model, tester, self.args.gpu)
        return best_test_bca, best_test_acc

    def _train_(self, w_l2, w_ur, random_state=None, val_size=0.2):
        self.model.rebuild_model(self.args.gpu)
        x_train, y_train, x_val, y_val = self.split_train_val(val_size, random_state)
        x_train, y_train = self.train_data

        optim = self.optim_func(self.model.parameters(), lr=self.args.lr)
        best_acc, count, best_test_acc, best_pos = 0, 0, 0, 0

        trainer = data_loader([x_train, y_train], True, batch_size=self.args.batch_size)
        valer = data_loader([x_val, y_val], True, batch_size=self.args.batch_size, shuffle=False)

        for e in range(self.args.epochs):
            self.model.train()
            start_t = time()
            for s, (inputs, labels) in enumerate(trainer):
                if self.args.gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                out, frs = self.model(inputs, with_frs=True)
                loss = self.loss_func(out, labels) + \
                       w_l2 * self.model.l2_loss() + \
                       w_ur * self.model.ur_loss(frs)

                optim.zero_grad()
                loss.backward()
                optim.step()
            val_acc = eval_acc(self.model, valer, self.args.gpu)
            end_t = time()
            print('\r[FLAG {:2d}][TRAIN {:4d}] Val ACC: {:.4f}, Best Val ACC: {:.4f}, Time: {:.2f}s'.format(
                self.args.flag, e, val_acc, best_acc, end_t - start_t), end='')
            if val_acc > best_acc:
                best_acc = val_acc
                count = 0
                t.save(self.model.state_dict(), self.save_path + '.tmp')
                best_pos = e
            else:
                count += 1
                if count > self.args.patience:
                    break
        self.model.load_state_dict(t.load(self.save_path + '.tmp'))
        best_bca = eval_bca(self.model, valer, self.args.gpu)
        return best_bca, best_acc, best_pos
