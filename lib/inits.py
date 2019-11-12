import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans
import os


def RPI(x, n_rules, sk=[25, 75]):
    in_dim = x.shape[1]
    n_sk = len(sk)
    skp = np.zeros([n_sk, in_dim])
    for i, s in enumerate(sk):
        skp[i] = np.percentile(x, sk[i], axis=0)
    Cs = np.zeros([in_dim, n_rules])
    for n in range(n_rules):
        Csidx = np.random.randint(0, n_sk, size=[in_dim])
        Cs[:, n] = np.choose(Csidx, skp)
    return Cs, None


def fcm_init(x_train, n_rules):
    n_samples, n_features = x_train.shape
    centers, mem, _, _, _, _, _ = fuzz.cmeans(
        x_train.T, n_rules, 2.0, error=1e-5, maxiter=200)
    delta = np.zeros([n_rules, n_features])
    for i in range(n_rules):
        d = (x_train - centers[i, :]) ** 2
        delta[i, :] = np.sum(d * mem[i, :].reshape(-1, 1), axis=0) / np.sum(mem[i, :])
    delta = np.sqrt(delta)
    delta = np.where(delta < 0.05, 0.05, delta)
    return centers.T, delta.T


def kmean_init(x_train, n_rules):
    Vs = np.ones([x_train.shape[1], n_rules])

    # if os.path.exists('ckpt/inits/{}_kmean_center.npz'.format(prefix)):
    #     print('[KMean Init] Loading from {}'.format(prefix))
    #     f = np.load('ckpt/inits/{}_kmean_center.npz'.format(prefix))
    #     Cs = f['Cs']
    # else:
    km = KMeans(n_rules, n_init=1)
    km.fit(x_train)
    Cs = km.cluster_centers_.T
    # np.savez('ckpt/inits/{}_kmean_center.npz'.format(prefix), Cs=km.cluster_centers_.T)
    return Cs, None


def kmean_init_save_load(x_train, n_rules, prefix='mnist'):
    Vs = np.ones([x_train.shape[1], n_rules])

    if os.path.exists('ckpt/inits/{}_kmean_center.npz'.format(prefix)):
        print('[KMean Init] Loading from {}'.format(prefix))
        f = np.load('ckpt/inits/{}_kmean_center.npz'.format(prefix))
        Cs = f['Cs']
    else:
        km = KMeans(n_rules, n_init=1)
        km.fit(x_train)
        Cs = km.cluster_centers_.T
        np.savez('ckpt/inits/{}_kmean_center.npz'.format(prefix), Cs=km.cluster_centers_.T)
    return Cs, None