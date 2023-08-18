import numpy as np
import copy
import torch
from tqdm import tqdm

def value_replace(w, value_sequence):  # w模型形式 ,value_sequence 数组形式，
    w_rel = copy.deepcopy(w)
    m =0
    print('-------Value Replacement------')
    for i in tqdm(w.keys()):
        for index, element in np.ndenumerate(w[i].cpu().numpy()): #顺序获取每一个值
            w_rel[i][index] = torch.tensor(value_sequence[m])
            m =m +1
    return w_rel


def defence_Trimmed_mean(beta, w):
    if isinstance(w[0], dict):
        w_list = []
        print('-----------Trimmmed-mean----------')
        for i in tqdm(range(len(w))):
            values_w = []
            for k in w[i].keys():
                values_w += list(w[i][k].view(-1).cpu().numpy())
            w_list.append(values_w)
        w_array = np.transpose(np.array(w_list))
        w_array.sort()

        w_list_sum = sum(np.array(w_list))
        w_min_sum = sum(np.transpose(w_array[: ,:beta]))  # 转置
        w_max_sum = sum(np.transpose(w_array[: ,-beta:]))

        w_avg_value = (w_list_sum - w_min_sum -w_max_sum ) /(len(w ) - 2 *beta)
        print(w_avg_value)
        w_avg = value_replace(w[0], w_avg_value)
    else:
        print('\nNot dict')

    return w_avg




