import torch
import numpy as np
import copy
import math
from tqdm import tqdm

def get_2_norm(params_a, params_b):
    sum = 0
    if isinstance(params_a,dict):
        for i in params_a.keys():
            if i.split('.')[-1] != 'num_batches_tracked':
                if len(params_a[i]) == 1:
                    sum += pow(np.linalg.norm(params_a[i].cpu().numpy()-\
                        params_b[i].cpu().numpy(), ord=2),2)
                else:
                    a = copy.deepcopy(params_a[i].cpu().numpy())
                    b = copy.deepcopy(params_b[i].cpu().numpy())
                    for j in range(len(a)):
                        x = copy.deepcopy(a[j].flatten())
                        y = copy.deepcopy(b[j].flatten())
                        sum += pow(np.linalg.norm(x-y, ord=2),2)
    else:
        sum += pow(np.linalg.norm(params_a-params_b, ord=2),2)
    norm = np.sqrt(sum)
    return norm

def defence_Krum(w, c):
    euclid_dist_list = []
    euclid_dist_matrix = [[0 for i in range(len(w))] for j in range(len(w))]
    for i in tqdm(range(len(w))):
        for j in range(i, len(w)):
            euclid_dist_matrix[i][j] = get_2_norm(w[i],w[j])
            euclid_dist_matrix[j][i] = euclid_dist_matrix[i][j]
        euclid_dist = euclid_dist_matrix[i][:]
        euclid_dist.sort()
        # print(sum(euclid_dist[:]))
        if len(w) >= (len(w)-c-2):
            euclid_dist_list.append(sum(euclid_dist[:len(w)-c-2]))
        else:
            euclid_dist_list.append(sum(euclid_dist))

    s_w = euclid_dist_list.index(min(euclid_dist_list))
    print('choosed index =',s_w)
    w_avg = w[s_w]
    return w_avg

def multi_krum(w,c):
    remaining_updates = w
    agg_w = []
    while len(remaining_updates) > 2*c +2:
        euclid_dist_list = []
        euclid_dist_matrix = [[0 for i in range(len(remaining_updates))] for j in range(len(remaining_updates))]
        for i in tqdm(range(len(remaining_updates))):
            for j in range(i, len(remaining_updates)):
                euclid_dist_matrix[i][j] = get_2_norm(remaining_updates[i], remaining_updates[j])
                euclid_dist_matrix[j][i] = euclid_dist_matrix[i][j]
            euclid_dist = euclid_dist_matrix[i][:]
            euclid_dist.sort()
            euclid_dist_list.append(sum(euclid_dist[:len(remaining_updates) - c - 2]))
        s_w = euclid_dist_list.index(min(euclid_dist_list))
        agg_w.append(remaining_updates[s_w])
        remaining_updates.pop(s_w)
    return agg_w