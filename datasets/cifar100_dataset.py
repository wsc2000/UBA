from torchvision import transforms
import numpy as np
from .backdoor_dataset import CIFAR10Pair, CIFAR10Pair_trust, CIFAR10Mem, BadEncoderTestBackdoor, BadEncoderDataset, ReferenceImg, BadEncoderDataset_union, BadEncoderTestBackdoor_v2, SERVERDETECT

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

finetune_transform_cifar10 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

backdoor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

classes = [
    'beaver', 'dolphin', 'otter', 'seal', 'whale',
    'aquarium fish', 'flatfish',' ray',' shark', 'trout',
    'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
    'bottles', 'bowls', 'cans', 'cups', 'plates',
    'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
    'clock', 'computer keyboard',' lamp', 'telephone', 'television',
    'bed', 'chair',' couch', 'table',' wardrobe',
    'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
    'bear',' leopard', 'lion', 'tiger', 'wolf',
    'bridge', 'castle',' house', 'road', 'skyscraper',
    'cloud', 'forest',' mountain',' plain',' sea',
    'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
    'fox',' porcupine', 'possum', 'raccoon',' skunk',
    'crab', 'lobster', 'snail',' spider', 'worm',
    'baby', 'boy', 'girl', 'man', 'woman',
    'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
    'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
    'maple', 'oak', 'palm', 'pine', 'willow',
    'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
    'lawn-mower', 'rocket', 'streetcar', 'tank',' tractor'
           ]


def get_fltrust_data_cifar100(data_dir):
    train_data = CIFAR10Pair_trust(numpy_file=data_dir, class_type=classes, transform=train_transform)
    return train_data

def get_detect_data_cifar100(data_dir):
    detect_data = SERVERDETECT(numpy_file=data_dir, transform = test_transform_cifar10)
    return detect_data


def get_pretraining_cifar100_1(data_dir, user_groups, idx):
    train_data = CIFAR10Pair(numpy_file=data_dir + "train.npz", class_type= classes, user_groups=user_groups, idx=idx, transform=train_transform)
    return train_data


def get_pretraining_cifar100_2(data_dir):
    memory_data = CIFAR10Mem(numpy_file=data_dir + "train.npz", class_type=classes, transform=test_transform_cifar10)
    test_data = CIFAR10Mem(numpy_file=data_dir + "test.npz", class_type=classes, transform=test_transform_cifar10)
    return memory_data, test_data


def get_detect_data_cifar100(data_dir):
    detect_data = SERVERDETECT(numpy_file=data_dir, transform = test_transform_cifar10)
    return detect_data

def get_shadow_cifar100(args, attacker, usergroups):
    # training_data_num = 50000
    # testing_data_num = 10000
    # np.random.seed(100)
    # training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)

    index = usergroups[attacker]
    index = np.asarray(index, dtype=int)

    print('loading from the training data')

    shadow_dataset = BadEncoderDataset(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.trigger_file,
        reference_file= args.reference_file,
        class_type=classes,
        indices = index,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )

    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor



def get_downstream_cifar100(args):
    training_file_name = 'train.npz'
    testing_file_name = 'test.npz'

    if args.encoder_usage_info == 'cifar10':
        print('test_transform_cifar10')
        test_transform = test_transform_cifar10
    elif args.encoder_usage_info == 'stl10':
        print('test_transform_stl10')
        test_transform = test_transform_stl10
    else:
        raise NotImplementedError

    target_dataset = ReferenceImg(reference_file=args.reference_file, transform=test_transform)
    memory_data = CIFAR10Mem(numpy_file=args.data_dir+training_file_name, class_type=classes, transform=test_transform)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+testing_file_name, trigger_file=args.trigger_file, reference_label= args.reference_label,  transform=test_transform)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+testing_file_name, class_type=classes, transform=test_transform)

    return target_dataset, memory_data, test_data_clean, test_data_backdoor



def get_shadow_cifar100_1(args, attacker, usergroups):
    # training_data_num = 50000
    # testing_data_num = 10000
    # np.random.seed(100)
    # training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)

    index = usergroups[attacker]
    index = np.asarray(index, dtype=int)

    print('loading from the training data')

    shadow_dataset = BadEncoderDataset_union(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.local_trigger1,
        reference_file= args.reference_file,
        class_type=classes,
        indices = index,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )

    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.local_trigger1, reference_label= args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor


def get_shadow_cifar100_2(args, attacker, usergroups):
    # training_data_num = 50000
    # testing_data_num = 10000
    # np.random.seed(100)
    # training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)

    index = usergroups[attacker]
    index = np.asarray(index, dtype=int)

    print('loading from the training data')

    shadow_dataset = BadEncoderDataset_union(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.local_trigger2,
        reference_file= args.reference_file,
        class_type=classes,
        indices = index,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )

    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.local_trigger2, reference_label= args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor


def get_shadow_cifar100_3(args, attacker, usergroups):
    # training_data_num = 50000
    # testing_data_num = 10000
    # np.random.seed(100)
    # training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)

    index = usergroups[attacker]
    index = np.asarray(index, dtype=int)

    print('loading from the training data')

    shadow_dataset = BadEncoderDataset_union(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.local_trigger3,
        reference_file= args.reference_file,
        class_type=classes,
        indices = index,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )

    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.local_trigger3, reference_label= args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor


def get_shadow_cifar100_4(args, attacker, usergroups):
    # training_data_num = 50000
    # testing_data_num = 10000
    # np.random.seed(100)
    # training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)

    index = usergroups[attacker]
    index = np.asarray(index, dtype=int)

    print('loading from the training data')

    shadow_dataset = BadEncoderDataset_union(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.local_trigger4,
        reference_file= args.reference_file,
        class_type=classes,
        indices = index,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )

    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor(numpy_file=args.data_dir+'test.npz', trigger_file=args.local_trigger4, reference_label= args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor



def get_shadow_cifar100_global(args, attacker, usergroups):
    # training_data_num = 50000
    # testing_data_num = 10000
    # np.random.seed(100)
    # training_data_sampling_indices = np.random.choice(training_data_num, training_data_num, replace=False)

    index = usergroups[attacker]
    index = np.asarray(index, dtype=int)

    print('loading from the training data')

    shadow_dataset = BadEncoderDataset_union(
        numpy_file=args.data_dir + 'train.npz',
        trigger_file=args.global_trigger_t,
        reference_file= args.reference_file,
        class_type=classes,
        indices = index,
        transform=train_transform,  # The train transform is not needed in BadEncoder.
        bd_transform=test_transform_cifar10,
        ftt_transform=finetune_transform_cifar10
    )

    memory_data = CIFAR10Mem(numpy_file=args.data_dir+'train.npz', class_type=classes, transform=test_transform_cifar10)
    test_data_backdoor = BadEncoderTestBackdoor_v2(numpy_file=args.data_dir+'test.npz', trigger_file=args.global_trigger_t, reference_label= args.reference_label,  transform=test_transform_cifar10)
    test_data_clean = CIFAR10Mem(numpy_file=args.data_dir+'test.npz', class_type=classes, transform=test_transform_cifar10)

    return shadow_dataset, memory_data, test_data_clean, test_data_backdoor