import torch
import copy
import numpy as np
from tqdm import tqdm
def average_weights_par(w, client_list):
    w_avg = copy.deepcopy(w[client_list[0]])
    if isinstance(w[0],dict) == True:
        for k in w_avg.keys():
            for i in range(1, len(client_list)):
                w_avg[k] += w[client_list[i]][k]
            w_avg[k] = torch.div(w_avg[k], len(client_list))
    else:
        for i in range(1, len(client_list)):
            w_avg += w[client_list[i]]
        w_avg = w_avg/len(client_list)
    return w_avg


def DNC(w, c, m, b):
    if isinstance(w[0] ,dict):
        w_list = []
        for i in tqdm(range(len(w))):
            values_w = []
            for k in w[i].keys():
                values_w += list(w[i][k].view(-1).cpu().numpy())
            if i == 0:
                sample_index = np.random.choice(range(len(values_w)), b)
                w_mean = np.array([values_w[j] for j in sample_index])
            else:
                w_mean += np.array([values_w[j] for j in sample_index])
            w_list.append([values_w[j] for j in sample_index])

        w_mean = w_mean /len(w)

        centr_matrix = []
        for i in range(len(w)):
            # centr_matrix.append((np.array(w_list[i] ) -w_mean).tolist())
            centr_matrix.append((np.array(w_list[i] ) -w_mean))

        U ,Sigma ,V = np.linalg.svd(centr_matrix ,full_matrices=False)

        outlier_score = []
        for i in range(len(w)):
            outlier_score.append(sum(np.dot(V, np.array(centr_matrix[i])))**2)

        id_sorted = np.argsort(outlier_score)

        set_G = [id_sorted[i] for i in range(int(len(w ) - c *m))]
        print('Choosed set is:',set_G)
        w_glob = average_weights_par(w, set_G)

    return w_glob