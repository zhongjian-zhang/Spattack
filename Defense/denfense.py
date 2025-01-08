import numpy as np
from collections import defaultdict


def median_sc(x, args):
    x = np.array(x)
    x = x.reshape(-1, args.dim)
    x = np.median(x, axis=0)
    return np.array(x)


def mean_sc(x, args):
    x = np.array(x)
    x = x.reshape(-1, args.dim)
    x = np.mean(x, axis=0)
    return np.array(x)


def sum_sc(x, args):
    x = np.array(x)
    x = x.reshape(-1, args.dim)
    x = np.sum(x, axis=0)
    return np.array(x)


def trimmedmean_sc(x, args):
    x = np.array(x)
    x = x.reshape(-1, args.dim)
    if args.clients_limit > 0:
        median = np.median(x, axis=0)
        tmp = x - median
        x = np.mean(
            np.take_along_axis(tmp, np.abs(tmp).argsort(axis=0)[:-int(args.m_cln_client * args.clients_limit), :],
                               axis=0),
            axis=0) + median
    else:
        x = np.mean(x, axis=0)
    return x


def krum_sc(x, args):
    x = np.array(x)
    x = x.reshape(-1, args.dim)
    distances = defaultdict(dict)
    non_malicious_count = x.shape[0]
    for i in range(x.shape[0]):
        for j in range(i):
            tmp = x[i] - x[j]
            distances[i][j] = distances[j][i] = np.linalg.norm(tmp)
    n_malicious = int(args.m_cln_client * args.clients_limit)
    if x.shape[0] > n_malicious + 2:
        non_malicious_count = x.shape[0] - n_malicious - 2
    elif x.shape[0] > n_malicious:
        non_malicious_count = x.shape[0] - n_malicious
    minimal_error = 1e20
    minimal_error_index = -1
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user
    return x[minimal_error_index]
