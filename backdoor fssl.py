import os
import copy
import numpy as np
# from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
import torch.nn.functional as F
import random
from PIL import Image
import pandas as pd
# from sklearn.preprocessing import StandardScaler
from models import get_encoder_architecture, get_encoder_architecture_usage
from datasets import get_usergroup, get_shadow_dataset, get_pretraining_dataset_1, get_pretraining_dataset_2, get_shadow_dataset1, get_shadow_dataset2 ,\
    get_shadow_dataset3, get_shadow_dataset4, get_shadow_dataset_global, get_server_detect, get_server_data
from evaluation import test
from models.update import average_weights
from models.krum_defense import defence_Krum,multi_krum
from models.trimmed_mean_defense import defence_Trimmed_mean
from models.median_defense import defense_median
from models.fltrust_defense import fltrust
from models.DNC_defence import DNC
from models.RFA import geometric_median_update
import matplotlib.pyplot as plt

def local_train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
        feature_1, out_1 = net(im_1)
        feature_2, out_2 = net(im_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)

        # compute loss （cosine similarity)
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        # loss = net(im_1, im_2, args)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Local Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.local_epoch, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num, net.state_dict()

def server_train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, im_2 in train_bar:
        im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)
        feature_1, out_1 = net(im_1)
        feature_2, out_2 = net(im_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.knn_t)

        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * 32, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * 32, -1)

        # compute loss （cosine similarity)
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.knn_t)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        # loss = net(im_1, im_2, args)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Local Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.local_epoch, train_optimizer.param_groups[0]['lr'], total_loss / total_num))

    return total_loss / total_num, net.state_dict()

def poisoning_train(backdoored_encoder, clean_encoder, data_loader, train_optimizer, poison_ep, args):
    backdoored_encoder.train()
    # freeze the BN layer
    for module in backdoored_encoder.modules():   # Returns an iterator over all modules in the network.
    # print(module)
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):    # Return whether the object has an attribute with the given name.
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    clean_encoder.eval()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_loss_0, total_loss_1, total_loss_2 = 0.0, 0.0, 0.0

    for img_clean, img_backdoor_list, reference_list, reference_aug_list in train_bar:

        img_clean = img_clean.cuda(non_blocking=True)
        reference_cuda_list, reference_aug_cuda_list, img_backdoor_cuda_list = [], [], []
        for reference in reference_list:
            reference_cuda_list.append(reference.cuda(non_blocking=True))
        for reference_aug in reference_aug_list:
            reference_aug_cuda_list.append(reference_aug.cuda(non_blocking=True))
        for img_backdoor in img_backdoor_list:
            img_backdoor_cuda_list.append(img_backdoor.cuda(non_blocking=True))

        clean_feature_reference_list = []

        with torch.no_grad():
            clean_feature_raw = clean_encoder(img_clean)
            clean_feature_raw = F.normalize(clean_feature_raw, dim=-1)
            for img_reference in reference_cuda_list:
                clean_feature_reference = clean_encoder(img_reference)
                clean_feature_reference = F.normalize(clean_feature_reference, dim=-1)
                clean_feature_reference_list.append(clean_feature_reference)

        feature_raw = backdoored_encoder(img_clean)
        feature_raw = F.normalize(feature_raw, dim=-1)

        feature_backdoor_list = []
        for img_backdoor in img_backdoor_cuda_list:
            feature_backdoor = backdoored_encoder(img_backdoor)
            feature_backdoor = F.normalize(feature_backdoor, dim=-1)
            feature_backdoor_list.append(feature_backdoor)

        feature_reference_list = []
        for img_reference in reference_cuda_list:
            feature_reference = backdoored_encoder(img_reference)
            feature_reference = F.normalize(feature_reference, dim=-1)
            feature_reference_list.append(feature_reference)

        feature_reference_aug_list = []
        for img_reference_aug in reference_aug_cuda_list:
            feature_reference_aug = backdoored_encoder(img_reference_aug)
            feature_reference_aug = F.normalize(feature_reference_aug, dim=-1)
            feature_reference_aug_list.append(feature_reference_aug)

        loss_0_list, loss_1_list = [], []
        for i in range(len(feature_reference_list)):
            loss_0_list.append(- torch.sum(feature_backdoor_list[i] * feature_reference_list[i], dim=-1).mean())  # 余弦相似度
            loss_1_list.append(- torch.sum(feature_reference_aug_list[i] * clean_feature_reference_list[i], dim=-1).mean())
        loss_2 = - torch.sum(feature_raw * clean_feature_raw, dim=-1).mean()

        loss_0 = sum(loss_0_list)/len(loss_0_list)
        loss_1 = sum(loss_1_list)/len(loss_1_list)

        loss = args.lambda1 * loss_0 + args.lambda1 * loss_1 + args.lambda2 * loss_2

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        total_loss_0 += loss_0.item() * data_loader.batch_size
        total_loss_1 += loss_1.item() * data_loader.batch_size
        total_loss_2 += loss_2.item() * data_loader.batch_size
        train_bar.set_description('Poisoning Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.6f}, Loss0: {:.6f}, Loss1: {:.6f},  Loss2: {:.6f}'.format(poison_ep, args.poison_epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num,  total_loss_0 / total_num , total_loss_1 / total_num,  total_loss_2 / total_num))

    return total_loss / total_num, backdoored_encoder.f.state_dict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Poisoning the local client to get a backdoor encoder')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr1', default=0.05, type=float, help='initial learning rate')
    parser.add_argument('--pretraining_dataset', type=str, default='stl10')

    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--lambda1', default=1.0, type=np.float64, help='value of labmda1')
    parser.add_argument('--lambda2', default=1.0, type=np.float64, help='value of labmda2')
    parser.add_argument('--poison_epochs', type=int, default=1, help='the number of poisoning epochs')
    parser.add_argument('--epochs', type=int, default=20, help="number of rounds of global training")
    parser.add_argument('--num_users', type=int, default= 25, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.4, help='the fraction of clients: C')
    parser.add_argument('--num_attackers', type=int, default=5, help="number of attackers")
    parser.add_argument('--local_epoch', type=int, default=2, help="the number of local epochs: E")
    parser.add_argument('--iid', type=int, default=0, help='Default set to IID. Set to 0 for non-IID.')

    parser.add_argument('--reference_file', default='./reference/stl10/airplane.npz', type=str,
                        help='path to the reference inputs')
    parser.add_argument('--shadow_dataset', default='stl10', type=str, help='shadow dataset')
    parser.add_argument('--encoder_usage_info', default='stl10', type=str,
                        help='used to locate encoder usage info, e.g., encoder architecture and input normalization parameter')
    parser.add_argument('--pretrained_encoder', default='./result/fed_pretrain_encoder/fed_pretrain_encoder_stl10200.pth', type=str,
                        help='path to the clean encoder used to finetune the backdoored encoder')

    parser.add_argument('--results_dir', default='./result/fed_badencoder', type=str, metavar='PATH',
                        help='path to save the backdoored encoder')

    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--gpu', default='0', type=str, help='which gpu the code runs on')

    parser.add_argument('--knn-t', default=0.5, type=float, help='softmax temperature in kNN monitor')
    parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')

    parser.add_argument('--global_trigger_t', default='./trigger/global_trigger.npz', type=str,
                        help='path to the global trigger')
    parser.add_argument('--local_trigger1', default='./trigger/global_trigger.npz', type=str,
                        help='path to the local trigger1')
    parser.add_argument('--local_trigger2', default='./trigger/global_trigger.npz', type=str,
                        help='path to the local trigger2')
    parser.add_argument('--local_trigger3', default='./trigger/global_trigger.npz', type=str,
                        help='path to the local trigger3')
    parser.add_argument('--local_trigger4', default='./trigger/global_trigger.npz', type=str,
                        help='path to the local trigger4')

    args = parser.parse_args()

    # Set the random seeds and GPU information
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.data_dir = f'./data/{args.pretraining_dataset}/'
    args.reference_label = 0
    print(args)

    user_groups = get_usergroup(args)  # type = dict, image indics for each client
    # print(user_groups.keys())

    # Training setup
    train_loss, train_accuracy = [], []

    attacker_list = list(set(np.random.choice(args.num_users, args.num_attackers, replace=False)))
    print('attacker list:', attacker_list)

    # for a in attacker_list:
    #     user_groups.pop(a)

    clean_model = get_encoder_architecture_usage(args).cuda()
    model = get_encoder_architecture_usage(args).cuda()
    detect_model = copy.deepcopy(clean_model)
    # Initialize the BadEncoder and load the pretrained encoder
    if args.pretrained_encoder != '':
        print(f'load the clean model from {args.pretrained_encoder}')
        if args.encoder_usage_info == 'cifar10':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
        elif args.encoder_usage_info == 'stl10':
            checkpoint = torch.load(args.pretrained_encoder)
            clean_model.load_state_dict(checkpoint['state_dict'])
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise NotImplementedError()

    global_model = copy.deepcopy(clean_model)
    detect_model = copy.deepcopy(clean_model)
    # model_mem = model
    # local_training_optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr, weight_decay=1e-6)

    results = {'BA': [], 'ASR_TEST': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print("==========================================================")
        local_weights, local_losses = [], []
        print(f'\n|Global Training Round : {epoch}|\n')
        m = max(int(args.frac * args.num_users), 0)

        attacker_list_ = []
        attacker_num_choice = 0
        """最少一个良性用户"""
        # m = max(int(args.frac * args.num_users), 1)

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print('All selected clients:',idxs_users)

        for a in attacker_list:
            attacker_id = 0
            if a in idxs_users:
                for i in idxs_users:
                    if i == a:
                        break
                    attacker_id += 1
                attacker_num_choice += 1
                attacker_list_.append(a)
                idxs_users = np.delete(idxs_users,[attacker_id])

        print('Benign clients:',idxs_users)
        print('The number of selected attackers:', attacker_num_choice)
        print('Selected attackers:',attacker_list_)
        local_weights_before = []

        global_model.train()

        for idx in idxs_users:
            train_data = get_pretraining_dataset_1(args, user_groups=user_groups, idx=idx)

            local_model = copy.deepcopy(global_model)
            local_weights_before.append(copy.deepcopy(local_model.state_dict()))
            local_training_optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=1e-6)

            train_loader = DataLoader(
                train_data,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True
            )
            print(f'--------Training process of client{idx}--------')
            for e in range(1, args.local_epoch + 1):

                loss, weights = local_train(local_model, train_loader, local_training_optimizer, e, args)

                local_losses.append(copy.deepcopy(loss))
            local_weights.append(copy.deepcopy(weights))

        num = 1
        if epoch % 1 == 0:
            for attacker in attacker_list_:
                model = copy.deepcopy(global_model)

                # Define the optimizer
                print("Fine-tune Optimizer: SGD")
                if args.encoder_usage_info == 'stl10':
                    finetune_optimizer = torch.optim.SGD(model.f.parameters(), lr=args.lr1, weight_decay=5e-4,
                                                         momentum=0.9)
                print(f'-!-!-!-!- Attacker{attacker}  Begin to poison the encoder -!-!-!-!-')
                if num == 1:
                    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset1(args, attacker, user_groups)
                elif num == 2:
                    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset2(args, attacker, user_groups)
                elif num == 3:
                    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset3(args, attacker, user_groups)
                elif num == 4:
                    shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset4(args, attacker, user_groups)
                num += 1

                test_loader_backdoor = DataLoader(
                    test_data_backdoor,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True)

                memory_loader = DataLoader(
                    memory_data,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )

                test_loader_clean = DataLoader(
                    test_data_clean,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )

                backdoor_train_loader = DataLoader(
                    shadow_data,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                    drop_last=True
                )

                """run normal training for 1 local epoch first"""
                normal_train_data = get_pretraining_dataset_1(args, user_groups, attacker)
                normal_loader = DataLoader(
                    normal_train_data,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                    drop_last=True
                )

                if epoch > 1:
                    print('local weight length:',len(local_weights))
                    print('local weight length_:',len(local_weights_))
                    if len(local_weights_) > 0:
                        # w = average_weights(local_weights_)
                        w = local_weights_[0]
                    else:
                        w = copy.deepcopy(global_model.state_dict())
                    model.load_state_dict(w,strict=False)
                local_weights_before.append(copy.deepcopy(model.state_dict()))

                for i in range(1):
                   _,_ = local_train(model, normal_loader, local_training_optimizer, i+1, args)

                for e in range(1, args.poison_epochs + 1):
                    if args.encoder_usage_info == 'stl10':
                        backdoored_loss, backdoored_weights = poisoning_train(model.f, clean_model.f, backdoor_train_loader, finetune_optimizer, e, args)
                    else:
                        raise NotImplementedError()

                # for i in range(1):
                #    _,_ = local_train(model, normal_loader, local_training_optimizer, i+1, args)

                model.load_state_dict(backdoored_weights, strict=False)
                local_weights.append(copy.deepcopy(model.state_dict()))
            local_weights_ = local_weights[10-attacker_num_choice:]
            print('local_weights_ length=========',len(local_weights_))

            update_w = average_weights(local_weights)

            if epoch >= 1:
                def get_grad(update, model):
                    '''get the update weight'''
                    grad = {}
                    for key, var in update.items():
                        grad[key] = update[key] - model[key]
                    return grad


                def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
                    vec = []
                    for key, param in net_dict.items():
                        # print(key, torch.max(param))
                        if key.split('.')[-1] == 'num_batches_tracked':
                            continue
                        vec.append(param.view(-1))
                    return torch.cat(vec)

                local_grad = []
                # local_grad_norm = []
                count = 0
                for w in local_weights_:
                    print('local weight before length:',len(local_weights_before))
                    local_grad.append(parameters_dict_to_vector_flt(get_grad(w, local_weights_before[10-attacker_num_choice+count])))
                    # local_grad_norm.append(torch.norm(parameters_dict_to_vector_flt(get_grad(w, local_weights_before[7+count]))))
                    count += 1
                global_g = get_grad(update_w, copy.deepcopy(global_model.state_dict()))
                global_g = parameters_dict_to_vector_flt(global_g)

                weight_measure = []
                for g in local_grad:
                    weight_measure.append(torch.cosine_similarity(g, global_g, dim=0))
                    # weight_measure.append((g - global_g))
                print('weight measure:',weight_measure)
                sim_sum = sum(weight_measure)
                weighted = []
                for i in weight_measure:
                    weighted.append(i / sim_sum)

                for w in weighted:
                    if w >= 10:
                        for i in range(len(weighted)):
                            weighted[i] = 1/attacker_num_choice
                        break
                print(weighted)

                if len(attacker_list_) == 5:
                    for key in local_weights_[0].keys():
                        local_weights_[0][key] = weighted[0] * local_weights_[0][key] + weighted[1] * local_weights_[1][
                            key] + weighted[2] * local_weights_[2][key] + weighted[3] * local_weights_[3][key]+ weighted[4] * local_weights_[4][key]
                    local_weights_[1] = local_weights_[0]
                    local_weights_[2] = local_weights_[0]
                elif len(attacker_list_) == 4:
                    for key in local_weights_[0].keys():
                        local_weights_[0][key] = weighted[0] * local_weights_[0][key] + weighted[1] * local_weights_[1][key] + weighted[2] * local_weights_[2][key] + weighted[3] * local_weights_[3][key]
                    local_weights_[1] = local_weights_[0]
                    local_weights_[2] = local_weights_[0]
                elif len(attacker_list_) == 3:
                    for key in local_weights_[0].keys():
                        local_weights_[0][key] = weighted[0] * local_weights_[0][key] + weighted[1] * local_weights_[1][key] + weighted[2] * local_weights_[2][key]
                    local_weights_[1] = local_weights_[0]
                    local_weights_[2] = local_weights_[0]
                elif len(attacker_list_) == 2:
                    for key in local_weights_[0].keys():
                        local_weights_[0][key] = weighted[0] * local_weights_[0][key] + weighted[1] * local_weights_[1][key]
                    local_weights_[1] = local_weights_[0]
                elif len(attacker_list_) == 1:
                    for key in local_weights_[0].keys():
                        local_weights_[0][key] = weighted[0] * local_weights_[0][key]

            global_model.load_state_dict(update_w)




        """Defense Mechanisms (Aggregation Rules)"""

        # """AVG Aggregator"""
        # update_w = average_weights(local_weights)
        # global_model.load_state_dict(update_w)

        # """gradients calculating"""
        # def get_grad(update, model):
        #     '''get the update weight'''
        #     grad = {}
        #     for key, var in update.items():
        #         grad[key] = update[key] - model[key]
        #     return grad
        #
        # global_para = global_model.state_dict()
        #
        # grad_list = []
        # for local_update in local_weights:
        #     grad_list.append(get_grad(local_update, global_para))

        # """Krum"""
        # print('==========Krum Aggregator==========')
        # update_w = defence_Krum(local_weights,3)
        # global_model.load_state_dict(update_w)

        # """Multi-Krum"""
        # print('==========Multi-Krum Aggregator==========')
        # agg_w = multi_krum(local_weights, 3)
        # update_w = average_weights(agg_w)
        # global_model.load_state_dict(update_w)
        #
        # """Trimmed-Mean"""
        # print('==========Trimmed-Mean Aggregator==========')
        # update_w = defence_Trimmed_mean(2, local_weights)
        # global_model.load_state_dict(update_w)

        # """Median"""
        # print('==========Median Aggregator==========')
        # updata_w = defense_median(local_weights)
        # global_model.load_state_dict(updata_w)

        # """DNC"""
        # print('==========DNC Aggregator==========')
        # update_w = DNC(local_weights, 1, 3, 10000)
        # global_model.load_state_dict(update_w)

        # """FLTrust"""
        # print('==========FLTrust Aggregator==========')
        # server_model = copy.deepcopy(global_model)
        # server_data = get_server_data(args, 'data/cifar10/server_data.npz')
        # server_loader = DataLoader(
        #     server_data,
        #     batch_size=32,
        #     shuffle=True,
        #     num_workers=2,
        #     pin_memory=True,
        #     drop_last=True
        # )
        # if args.encoder_usage_info == 'cifar10':
        #     for ep in range(3):
        #         _, fltrust_norm = server_train(server_model, server_loader, local_training_optimizer, ep + 1, args)
        # else:
        #     for ep in range(10):
        #         _, fltrust_norm = server_train(server_model, server_loader, local_training_optimizer, ep + 1, args)
        # fltrust_norm = get_grad(fltrust_norm, global_para)
        # aggregate_weight = fltrust(grad_list, fltrust_norm, global_para)
        # global_model.load_state_dict(aggregate_weight)

        # """RFA"""
        # print('==========RFA Aggregator==========')
        # alpha = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        # update_w = geometric_median_update(local_weights, alpha)
        # global_model.load_state_dict(update_w)

        # """Our EmbInspector"""
        # print('========== EmbInspector ==========')
        #
        # # Creating detecting data
        # detect_data_path = 'data/cifar100/server_data_10.npz'
        # server_detect_data = get_server_detect(args, detect_data_path)
        # detecet_loader = DataLoader(
        #     server_detect_data,
        #     batch_size=1,
        #     shuffle=True,
        #     # num_workers=2,
        #     pin_memory=True,
        #     drop_last=True
        # )
        #
        # detecet_feature_set = []
        # for weight in local_weights:
        #     local_detecet_feature_list = []
        #     detect_model.load_state_dict(weight)
        #     detect_model.eval()
        #
        #     for detect_img in tqdm(detecet_loader):
        #         detect_img = detect_img.cuda(non_blocking=True)
        #         feature = detect_model.f(detect_img)
        #         feature = F.normalize(feature, dim=-1)
        #         local_detecet_feature_list.append(feature)
        #     # print(len(local_detecet_feature_list))
        #     detecet_feature_set.append(local_detecet_feature_list)
        #
        # cosine_similarity_list = []
        #
        # # initialize the malicious score
        # malicious_score = []
        # for i in range(len(detecet_feature_set)):
        #     malicious_score.append(0)
        #
        # print('=========== Malicious scores computing ==========')
        # # caculate the cosine simlarity
        # for detect_iter in tqdm(range(len(local_detecet_feature_list))):  # iterate for len(detect images)
        #     detecet_list = []
        #     for c in range(len(detecet_feature_set)): # compute pair-wise cos sim distance
        #         similarity = 0
        #         for client in range(len(detecet_feature_set)):
        #             # cosine similarity
        #             similarity += torch.sum(detecet_feature_set[c][detect_iter] * detecet_feature_set[client][detect_iter], dim=-1).mean()
        #             # L2 distance
        #             # list1_temp = [item.cpu().detach().numpy() for item in detecet_feature_set[c][detect_iter]]
        #             # list2_temp = [item.cpu().detach().numpy() for item in detecet_feature_set[client][detect_iter]]
        #             # list_temp = [list1_temp[0],list2_temp[0]]
        #             # print(np.asarray(list_temp).shape)
        #             # similarity -= torch.pdist(torch.tensor(list_temp), p=2).mean()
        #         similarity = similarity.cpu().detach().numpy()
        #         detecet_list.append(similarity)
        #
        #     detecet_list = np.asarray(detecet_list).reshape(-1, 1)
        #
        #     copy_list = copy.deepcopy(detecet_list)
        #
        #     list_len = len(copy_list)
        #     if list_len % 2 == 0:
        #         sim_median = (copy_list[int(list_len / 2)] + copy_list[int(list_len / 2 - 1)]) / 2
        #     else:
        #         sim_median = copy_list[int((list_len + 1) / 2 - 1)]
        #
        #     # compute avg
        #     sim_avg = np.mean(detecet_list)
        #
        #     decision_boundary = max(sim_avg, sim_median)
        #
        #     client_index = 0
        #     for sim in detecet_list:
        #         if sim >= decision_boundary:
        #             malicious_score[client_index] += 1
        #         else:
        #             malicious_score[client_index] -= 1
        #         client_index += 1
        #
        # print('malicious scores = ',malicious_score)
        # malicious_clients_index = []
        # client_index = 0
        # for s in malicious_score:
        #     if s > 0:
        #         malicious_clients_index.append(client_index)
        #     client_index += 1
        #
        # print('malicious client are:',malicious_clients_index)
        # malicious_clients_index.reverse()
        #
        # for attacker in malicious_clients_index:
        #     local_weights.pop(attacker)
        #
        # if len(local_weights) == 0:
        #     global_model = copy.deepcopy(global_model)
        # else:
        #     global_weights = average_weights(local_weights)
        #     global_model.load_state_dict(global_weights)



        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        print(f'Average loss of all clients:{loss_avg}')
        print('-----Global model clean & backdoored test-----:')

        shadow_data, memory_data, test_data_clean, test_data_backdoor = get_shadow_dataset_global(args, 0, user_groups)
        test_loader_backdoor = DataLoader(
            test_data_backdoor,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        memory_loader = DataLoader(
            memory_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        test_loader_clean = DataLoader(
            test_data_clean,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print('================global model test=================')
        test_global_ba_acc, ba, asr_test = test(global_model.f, memory_loader, test_loader_clean, test_loader_backdoor, epoch,
                                  args)

        results['BA'].append(ba)
        results['ASR_TEST'].append(asr_test)
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(args.results_dir + '/SBA.csv', index_label='epoch')

        if epoch % args.epochs == 0:
            torch.save(
                {'epoch': epoch, 'state_dict': global_model.state_dict(),
                 'localtraining_optimizer': local_training_optimizer.state_dict(),
                 'finetune_optimizer': finetune_optimizer.state_dict(),
                },
                 args.results_dir + '/cifar100_backdoor_camel' + str(epoch) + '.pth'
                     )
