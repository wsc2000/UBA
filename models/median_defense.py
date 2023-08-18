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


def defense_median(w):
    if isinstance(w[0],dict):
        w_list = []
        print('-------Median-------')
        for i in tqdm(range(len(w))):
            values_w = []
            for k in w[i].keys():
                values_w += list(w[i][k].view(-1).cpu().numpy())
            w_list.append(values_w)
        w_array = np.transpose(np.array(w_list))
        # print(w_array)
        w_array.sort()
        # print(w_array)
        # print(len(w_array))
        # print(w_array.shape)
        if len(w)%2 != 0:
            index = int((len(w)-1)/2)
            print('index=',index)
            w_median_value = w_array[:,index]
        else:
            index1 = int((len(w)/2))
            index2 = int((len(w)/2) - 1)
            print('index=', index1,',',index2)
            w_median_value = (w_array[:,index1] + w_array[:,index2])/2
        print('w_median_value=',w_median_value)
        w_median = value_replace(w[0], w_median_value)

        return w_median