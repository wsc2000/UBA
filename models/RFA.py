import numpy as np
import torch
import copy
from tqdm import tqdm

def weighted_average_oracle(points, weights):
    """
    Computes weighted average of atoms with specified weights
    """
    w_avg = copy.deepcopy(points[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * weights[0]
        for i in range(1, len(points)):
            w_avg[key] += points[i][key] * weights[i]

    return w_avg

def l2dist(params_a, params_b):
    sum = 0
    if isinstance(params_a,dict):
        for i in params_a.keys():
            if i.split('.')[-1] != 'num_batches_tracked':
                if len(params_a[i]) == 1:
                    sum += np.linalg.norm(params_a[i].cpu().numpy()-\
                        params_b[i].cpu().numpy(), ord=2)
                else:
                    a = copy.deepcopy(params_a[i].cpu().numpy())
                    b = copy.deepcopy(params_b[i].cpu().numpy())
                    for j in range(len(a)):
                        x = copy.deepcopy(a[j].flatten())
                        y = copy.deepcopy(b[j].flatten())
                        sum += np.linalg.norm(x-y, ord=2)
    else:
        sum += np.linalg.norm(params_a-params_b, ord=2)

    return sum

def geometric_median_objective(median, points, alphas):
    """Compute geometric median objective."""
    return sum([alpha * l2dist(median, p) for alpha, p in zip(alphas, points)])

def geometric_median_update(points, alphas, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6):
    """
    Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
    """
    # alphas = np.asarray(alphas, dtype=points[0].dtype) / sum(alphas)
    alphas = np.asarray(alphas) / sum(alphas)
    median = weighted_average_oracle(points, alphas)
    # num_oracle_calls = 1

    # logging
    obj_val = geometric_median_objective(median, points, alphas)
    logs = []
    log_entry = [0, obj_val, 0, 0]
    logs.append(log_entry)
    if verbose:
        print('Starting Weiszfeld algorithm')
        print(log_entry)

    # start
    for i in range(maxiter):
        prev_median, prev_obj_val = median, obj_val
        weights = np.asarray([alpha / max(eps,l2dist(median, p)) for alpha, p in zip(alphas, points)],
                             dtype=alphas.dtype)
        weights = weights / weights.sum()
        print('weights list is ',weights)
        median = weighted_average_oracle(points, weights)
        # num_oracle_calls += 1
        obj_val = geometric_median_objective(median, points, alphas)
        log_entry = [i+1, obj_val,
                     (prev_obj_val - obj_val)/obj_val,
                     l2dist(median, prev_median)]
        logs.append(log_entry)
        if verbose:
            print(log_entry)
        if abs(prev_obj_val - obj_val) < ftol * obj_val:
            break
    return median

