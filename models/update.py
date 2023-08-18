"""FedAvg Aggregation"""
import numpy as np
import torch
import copy

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            # w_avg[key] += w[i][key]
            w_avg[key] = w_avg[key] + w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def weighted_aggregating(w,weight):

    w_avg = copy.deepcopy(w[0])

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + weight[i] * w[i][key]
        w_avg[key] = w_avg[key] - w[0][key] + weight[0] * w[0][key]


    # for i in range(len(w)):
    #     w_avg = w_avg + w[i] * weight[i]


    return w_avg