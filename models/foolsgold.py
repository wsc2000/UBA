import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
from tqdm import tqdm

logger = logging.getLogger('logger')

def save_history(userID = 0):
    folderpath = '{0}/foolsgold'.format(f'saved_models')
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    history_name = '{0}/history_{1}.pth'.format(folderpath, userID)
    update_name = '{0}/saved_updates/update_{1}.pth'.format(f'saved_models', userID)


    model = torch.load(update_name)
    if os.path.exists(history_name):
        loaded_params = torch.load(history_name)
        history = dict()
        for name, data in loaded_params.items():
            # if True:
            #     continue
            history[name] = data + model[name]
        torch.save(history, history_name)
    else:
        torch.save(model, history_name)

def foolsgold_aggr(weight_accumulator, current_users):
    for i in current_users:
        save_history(userID = i)

    layer_name = 'g.3'
    epsilon = 1e-5
    folderpath = '{0}/foolsgold'.format(f'saved_models')
    # Load params
    his = []
    for i in current_users:
        history_name = '{0}/history_{1}.pth'.format(folderpath, i)
        his_i_params = torch.load(history_name)
        for name, data in his_i_params.items():
            # his_i = np.append(his_i, ((data.cpu().numpy()).flatten()))
           if layer_name in name:
                his = np.append(his, (data.cpu().numpy()).flatten())
    his = np.reshape(his, (len(current_users), -1))
    logger.info("FoolsGold: Finish loading history updates")
    cs = smp.cosine_similarity(his) - np.eye(len(current_users))

    # print(cs)

    maxcs = np.max(cs, axis=1) + epsilon

    # print(maxcs)
    for i in tqdm(range(len(current_users)),desc='FoolsGold'):
        for j in range(len(current_users)):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    logger.info("FoolsGold: Calculate max similarities")
    # Pardoning
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0


    if np.max(wv) <= 0:
        wv = [1, 0, 0, 0, 0, 0,0, 0, 0, 0]

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    # wv = np.asarray(wv,dtype=np.float32)



    # Federated SGD iteration
    logger.info(f"FoolsGold: Accumulation with lr {wv}")

    wv = list(wv)
    print(wv)

    client_sq = 0
    for i in current_users:
        update_name = '{0}/saved_updates/update_{1}.pth'.format(f'saved_models', i)
        update_params = torch.load(update_name)
        for name, data in update_params.items():
            # if True:
            #     continue
            # print(type(wv[client_sq]))
            # print(type(data))
            weight_accumulator[name] = torch.tensor(weight_accumulator[name], dtype=torch.float32)
            weight_accumulator[name] += (wv[client_sq] * data).to(torch.device('cuda'))
        client_sq += 1
    return weight_accumulator