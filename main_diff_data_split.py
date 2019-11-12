import argparse
import os

import scipy.io as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from lib.inits import *
from lib.models import *
from lib.tuning_train import *

np.random.seed(1447)
t.manual_seed(1447)


def get_parser():
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--gpu', dest='gpu', action='store_true')
    flag_parser.add_argument('--cpu', dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)

    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--bn', dest='bn', action='store_true')
    flag_parser.add_argument('--no_bn', dest='bn', action='store_false')
    parser.set_defaults(bn=True)

    # training parameters
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=1000, type=int, help='total training epochs')
    parser.add_argument('--patience', default=40, type=int, help='training patience')
    parser.add_argument('--data', default='Abalone', type=str, help='using dataset')
    parser.add_argument('--n_rules', default=20, type=int, help='number of rules')
    parser.add_argument('--loss_type', default='crossentropy', type=str, help='using loss')
    parser.add_argument('--optim_type', default='adabound', type=str, help='type of optimization')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='l2 loss weight')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')

    # weights of regularization
    parser.add_argument('--weight_frs', default=0, type=float, help='ur loss weight')

    parser.add_argument('--init', default='kmean', type=str, help='')
    parser.add_argument('--tune_param', default=1, type=int, help='whether to tune parameter')
    parser.add_argument('--repeats', default=1, type=int, help='repeat to get best pos')

    return parser.parse_args()


parser = argparse.ArgumentParser()
args = get_parser()
data_root = 'data/'


def run(flag, tail):
    args.flag = flag
    save_path = 'diff_split/ckpt/{}_{}_{}.pkl'.format(args.data, flag, tail)

    f = np.load(os.path.join(data_root, args.data + '.npz'))
    if flag == 0:
        print('Loading {} data, saving to {}'.format(args.data, save_path))
    data = f['con_data'].astype('float')
    label = f['label']
    n_classes = len(np.unique(label))
    train_idx, test_idx = f['trains'][flag], f['tests'][flag]

    x_train, y_train = data[train_idx], label[train_idx]
    x_test, y_test = data[test_idx], label[test_idx]

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    n_rules = args.n_rules

    if args.init == 'rpi':
        Cs, Vs = RPI(x_train, n_rules)
    elif args.init == 'kmean':
        Cs, Vs = kmean_init(x_train, n_rules)
    elif args.init == 'fcm':
        Cs, Vs = fcm_init(x_train, n_rules)
    else:
        exit()

    model = ClsTSK(x_train.shape[1], n_rules, n_classes, init_centers=Cs, init_Vs=Vs, bn=args.bn)
    Train = ClassModelTrain(
        model=model, train_data=(x_train, y_train),
        test_data=(x_test, y_test), n_classes=n_classes,
        args=args, save_path=save_path, optim_type=args.optim_type
    )
    best_test_bca, best_test_acc = Train.train()
    print('[FLAG {:2d}] ACC: {:.4f}, BCA: {:.4f}'.format(flag, best_test_acc, best_test_bca))
    return best_test_acc, best_test_bca


if __name__ == '__main__':
    n_repeats = 30
    hist = [1] * n_repeats
    best_acc = [0] * n_repeats
    best_bca = [0] * n_repeats
    tail = args.loss_type
    if args.weight_frs > 0:
        tail += '_ur'
    if not args.bn:
        tail += '_noBN'
    if args.init != 'kmean':
        tail += '_{}'.format(args.init)
    save_path = 'diff_split/res/{}_{}.mat'.format(args.data, tail)
    if os.path.exists(save_path):
        print('{} Exists.'.format(save_path))
        exit()
    print('saving to {}'.format(save_path))
    for i in range(n_repeats):
        best_acc[i], best_bca[i] = run(i, tail)
    print(save_path, np.mean(best_acc), np.mean(best_bca))
    sp.savemat(save_path, {'best_acc': best_acc, 'best_bca': best_bca})
