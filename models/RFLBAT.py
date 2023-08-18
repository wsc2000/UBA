import math
import random
from copy import deepcopy
from typing import List, Any, Dict
import torch
from torch import nn
import logging
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.metrics.pairwise as smp
from models.update import average_weights
import warnings
logger = logging.getLogger('logger')

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
def gap_statistics(data, num_sampling, K_max, n):
    num_cluster = 0
    data = np.reshape(data, (data.shape[0], -1))
    # Linear transformation
    data_c = np.ndarray(shape=data.shape)
    for i in range(data.shape[1]):
        data_c[:, i] = (data[:, i] - np.min(data[:, i])) / \
                       (np.max(data[:, i]) - np.min(data[:, i]))
    gap = []
    s = []
    for k in range(1, K_max + 1):
        k_means = KMeans(n_clusters=k, init='k-means++').fit(data_c)
        predicts = (k_means.labels_).tolist()
        centers = k_means.cluster_centers_
        v_k = 0
        for i in range(k):
            for predict in predicts:
                if predict == i:
                    v_k += np.linalg.norm(centers[i] - \
                                          data_c[predicts.index(predict)])
        # perform clustering on fake data
        v_kb = []
        for _ in range(num_sampling):
            data_fake = []
            for i in range(n):
                temp = np.ndarray(shape=(1, data.shape[1]))
                for j in range(data.shape[1]):
                    temp[0][j] = random.uniform(0, 1)
                data_fake.append(temp[0])
            k_means_b = KMeans(n_clusters=k, init='k-means++').fit(data_fake)
            predicts_b = (k_means_b.labels_).tolist()
            centers_b = k_means_b.cluster_centers_
            v_kb_i = 0
            for i in range(k):
                for predict in predicts_b:
                    if predict == i:
                        v_kb_i += np.linalg.norm(centers_b[i] - \
                                                 data_fake[predicts_b.index(predict)])
            v_kb.append(v_kb_i)
        # gap for k
        v = 0
        for v_kb_i in v_kb:
            # print(v_kb_i)
            if v_kb_i == 0:
                continue
            v += math.log(v_kb_i)
        v /= num_sampling
        gap.append(v - math.log(v_k))
        sd = 0
        for v_kb_i in v_kb:
            if v_kb_i == 0:
                continue
            sd += (math.log(v_kb_i) - v) ** 2
        sd = math.sqrt(sd / num_sampling)
        s.append(sd * math.sqrt((1 + num_sampling) / num_sampling))
    # select smallest k
    for k in range(1, K_max + 1):
        print(gap[k - 1] - gap[k] + s[k - 1])
        if k == K_max:
            num_cluster = K_max
            break
        if gap[k - 1] - gap[k] + s[k - 1] > 0:
            num_cluster = k
            break
    return num_cluster

def RFLBAT(gradients, weights, _):
    eps1 = 10
    eps2 = 4
    dataAll = gradients
    # for i in range(len(gradients)):
    #     # file_name = '{0}/saved_updates/update_{1}.pth'\
    #     #     .format(self.params.folder_path,i)
    #     dataList = []
    #     if os.path.exists(file_name):
    #         loaded_params = torch.load(file_name)
    #         for name, data in loaded_params.items():
    #             if 'MNIST' in self.params.task or \
    #                 'fc' in name or 'layer4.1.conv' in name:
    #                 dataList.extend(((data.cpu().numpy())\
    #                     .flatten()).tolist())
    #         dataAll.append(dataList)
    pca = PCA(n_components=2) #instantiate
    pca = pca.fit(dataAll)
    X_dr = pca.transform(dataAll)

    # Save figure
    # plt.figure()
    # plt.scatter(X_dr[0:self.params.fl_number_of_adversaries,0],
    #     X_dr[0:self.params.fl_number_of_adversaries,1], c='red')
    # plt.scatter(X_dr[self.params.fl_number_of_adversaries:self.params.fl_total_participants,0],
    #     X_dr[self.params.fl_number_of_adversaries:self.params.fl_total_participants,1], c='green')
    # # plt.scatter(X_dr[self.params.fl_total_participants:,0], X_dr[self.params.fl_total_participants:,1], c='black')
    # folderpath = '{0}/RFLBAT'.format(self.params.folder_path)
    # if not os.path.exists(folderpath):
    #     os.makedirs(folderpath)
    # figname = '{0}/PCA_E{1}.jpg'.format(folderpath, self.current_epoch)
    # plt.savefig(figname)
    # logger.info(f"RFLBAT: Save figure {figname}.")

    # Compute sum eu distance
    eu_list = []
    for i in range(len(X_dr)):
        eu_sum = 0
        for j in range(len(X_dr)):
            if i==j:
                continue
            eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
        eu_list.append(eu_sum)
    accept = []
    x1 = []
    for i in range(len(eu_list)):
        if eu_list[i] < eps1 * np.median(eu_list):
            accept.append(i)
            x1 = np.append(x1, X_dr[i])
        else:
            logger.info("RFLBAT: discard update {0}".format(i))
    x1 = np.reshape(x1, (-1, X_dr.shape[1]))
    num_clusters = gap_statistics(x1, \
        num_sampling=5, K_max=10, n=len(x1))
    logger.info("RFLBAT: the number of clusters is {0}"\
        .format(num_clusters))
    k_means = KMeans(n_clusters=num_clusters, \
        init='k-means++').fit(x1)
    predicts = k_means.labels_

    # select the most suitable cluster
    v_med = []
    for i in range(num_clusters):
        temp = []
        for j in range(len(predicts)):
            if predicts[j] == i:
                temp.append(dataAll[accept[j]])
        if len(temp) <= 1:
            v_med.append(1)
            continue
        v_med.append(np.median(np.average(smp\
            .cosine_similarity(temp), axis=1)))
    temp = []
    for i in range(len(accept)):
        if predicts[i] == v_med.index(min(v_med)):
            temp.append(accept[i])
    accept = temp

    # compute eu list again to exclude outliers
    temp = []
    for i in accept:
        temp.append(X_dr[i])
    X_dr = temp
    eu_list = []
    for i in range(len(X_dr)):
        eu_sum = 0
        for j in range(len(X_dr)):
            if i==j:
                continue
            eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
        eu_list.append(eu_sum)
    temp = []
    for i in range(len(eu_list)):
        if eu_list[i] < eps2 * np.median(eu_list):
            temp.append(accept[i])
        else:
            logger.info("RFLBAT: discard update {0}"\
                .format(i))
    accept = temp
    logger.info("RFLBAT: the final clients accepted are {0}"\
        .format(accept))

    weights_for_agg =[]
    print('Accepted clients:',accept)

    # aggregate
    for i in range(len(gradients)):
        if i in accept:
            weights_for_agg.append(weights[i])

    global_w = average_weights(weights_for_agg)

    return global_w

    #         update_name = '{0}/saved_updates/update_{1}.pth'\
    #             .format(self.params.folder_path, i)
    #         loaded_params = torch.load(update_name)
    #         self.accumulate_weights(weight_accumulator,
    #             {key:loaded_params[key].to(self.params.device) for key \
    #             in loaded_params})
    # self.current_epoch += 1