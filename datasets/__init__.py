from .cifar10_dataset import get_pretraining_cifar10_1, get_pretraining_cifar10_2, get_shadow_cifar10, get_downstream_cifar10,\
    get_shadow_cifar10_1, get_shadow_cifar10_2, get_shadow_cifar10_3, get_shadow_cifar10_4, get_shadow_cifar10_global, get_detect_data_cifar10,\
    get_fltrust_data_cifar10
from .sampling import cifar_iid, cifar_noniid
from .stl10_dataset import get_downstream_stl10, get_pretraining_stl10_1, get_pretraining_stl10_2, get_shadow_stl10_1,get_shadow_stl10_2,\
    get_shadow_stl10_3, get_shadow_stl10_4, get_shadow_stl10_global, get_detect_data_stl10, get_shadow_stl10, get_fltrust_data_stl10
from .cifar100_dataset import get_pretraining_cifar100_1, get_pretraining_cifar100_2, get_shadow_cifar100, get_downstream_cifar100,\
    get_shadow_cifar100_1, get_shadow_cifar100_2, get_shadow_cifar100_3, get_shadow_cifar100_4, get_shadow_cifar100_global,\
    get_detect_data_cifar100, get_fltrust_data_cifar100
from .gtsrb_dataset import get_downstream_gtsrb

import numpy as np

def get_server_data(args,data_dir):
    if args.pretraining_dataset == 'cifar10':
        return get_fltrust_data_cifar10(data_dir)
    elif args.pretraining_dataset == 'cifar100':
        return get_fltrust_data_cifar100(data_dir)
    if args.pretraining_dataset =='stl10':
        return get_fltrust_data_stl10(data_dir)


def get_server_detect(args, data_dir):
    if args.pretraining_dataset == 'cifar10':
        return get_detect_data_cifar10(data_dir)
    elif args.pretraining_dataset == 'stl10':
        return get_detect_data_stl10(data_dir)
    elif args.pretraining_dataset == 'cifar100':
        return get_detect_data_cifar100(data_dir)

def get_pretraining_dataset_1(args, user_groups, idx):
    if args.pretraining_dataset == 'cifar10':
        return get_pretraining_cifar10_1(args.data_dir, user_groups, idx)
    elif args.pretraining_dataset == 'cifar100':
        return get_pretraining_cifar100_1(args.data_dir, user_groups, idx)
    elif args.pretraining_dataset == 'stl10':
        return get_pretraining_stl10_1(args.data_dir, user_groups, idx)
    else:
        raise NotImplementedError

def get_pretraining_dataset_2(args):
    if args.pretraining_dataset == 'cifar10':
        return get_pretraining_cifar10_2(args.data_dir)
    elif args.pretraining_dataset == 'cifar100':
        return get_pretraining_cifar100_2(args.data_dir)
    elif args.pretraining_dataset == 'stl10':
        return get_pretraining_stl10_2(args.data_dir)
    else:
        raise NotImplementedError


def get_usergroup(args):
    if args.pretraining_dataset == 'stl10':
        train_data = np.load(f'./data/{args.pretraining_dataset}/train_unlabeled.npz')
    else:
        train_data = np.load(f'./data/{args.pretraining_dataset}/train.npz')
    if args.iid:
        user_groups = cifar_iid(train_data['x'], args.num_users)
    else:
        user_groups = cifar_noniid(train_data, args.num_users)

    return user_groups


# every poison epoch needs to use this method
def get_shadow_dataset(args, attackers, user_groups):
    if args.shadow_dataset == 'cifar10':
        return get_shadow_cifar10(args, attackers, user_groups)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10(args, attackers, user_groups)
    elif args.shadow_dataset == 'cifar100':
        return get_shadow_cifar100(args, attackers, user_groups)
    else:
        raise NotImplementedError

def get_dataset_evaluation(args):
    if args.dataset =='cifar10':
        return get_downstream_cifar10(args)
    elif args.dataset == 'cifar100':
        return get_downstream_cifar100(args)
    elif args.dataset == 'stl10':
        return get_downstream_stl10(args)
    elif args.dataset == 'gtsrb':
        return get_downstream_gtsrb(args)
    else:
        raise NotImplementedError


def get_shadow_dataset1(args, attackers, user_groups):
    if args.shadow_dataset == 'cifar10':
        return get_shadow_cifar10_1(args, attackers, user_groups)
    elif args.shadow_dataset == 'cifar100':
        return get_shadow_cifar100_1(args, attackers, user_groups)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10_1(args, attackers, user_groups)
    else:
        raise NotImplementedError

def get_shadow_dataset2(args, attackers, user_groups):
    if args.shadow_dataset == 'cifar10':
        return get_shadow_cifar10_2(args, attackers, user_groups)
    elif args.shadow_dataset == 'cifar100':
        return get_shadow_cifar100_2(args, attackers, user_groups)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10_2(args, attackers, user_groups)
    else:
        raise NotImplementedError


def get_shadow_dataset3(args, attackers, user_groups):
    if args.shadow_dataset == 'cifar10':
        return get_shadow_cifar10_3(args, attackers, user_groups)
    elif args.shadow_dataset == 'cifar100':
        return get_shadow_cifar100_3(args, attackers, user_groups)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10_3(args, attackers, user_groups)
    else:
        raise NotImplementedError

def get_shadow_dataset4(args, attackers, user_groups):
    if args.shadow_dataset == 'cifar10':
        return get_shadow_cifar10_4(args, attackers, user_groups)
    elif args.shadow_dataset == 'cifar100':
        return get_shadow_cifar100_4(args, attackers, user_groups)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10_4(args, attackers, user_groups)
    else:
        raise NotImplementedError


def get_shadow_dataset_global(args, attackers, user_groups):
    if args.shadow_dataset == 'cifar10':
        return get_shadow_cifar10_global(args, attackers, user_groups)
    elif args.shadow_dataset == 'cifar100':
        return get_shadow_cifar100_global(args, attackers, user_groups)
    elif args.shadow_dataset == 'stl10':
        return get_shadow_stl10_global(args, attackers, user_groups)
    else:
        raise NotImplementedError
