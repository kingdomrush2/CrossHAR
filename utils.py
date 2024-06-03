import argparse
import torch.nn as nn
import random
import numpy as np
import torch
import sys
import itertools
import matplotlib.pyplot as plt

from scipy.special import factorial
from torch.utils.data import Dataset
from config import create_io_config, load_dataset_stats, TrainConfig, MaskConfig, load_model_config
from transforms3d.axangles import axangle2mat
from augmentations import DataTransform
from itertools import permutations
from scipy.stats import special_ortho_group

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device(gpu):
    if gpu is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("%s (%d GPUs)" % (device, n_gpu))
    return device


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def bert_mask(seq_len, goal_num_predict):
    return random.sample(range(seq_len), goal_num_predict)


def span_mask(seq_len, max_gram=3, p=0.2, goal_num_predict=15):
    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = p * np.power(1 - p, np.arange(max_gram))
    # alpha = 6
    # pvals = np.power(alpha, ngrams) * np.exp(-alpha) / factorial(ngrams)# possion
    pvals /= pvals.sum(keepdims=True)
    mask_pos = set()
    while len(mask_pos) < goal_num_predict:
        n = np.random.choice(ngrams, p=pvals)
        n = min(n, goal_num_predict - len(mask_pos))
        anchor = np.random.randint(seq_len)
        if anchor in mask_pos:
            continue
        for i in range(anchor, min(anchor + n, seq_len - 1)):
            mask_pos.add(i)
    return list(mask_pos)


def merge_dataset(data, label, mode='all'):
    index = np.zeros(data.shape[0], dtype=bool)
    label_new = []
    for i in range(label.shape[0]):
        if mode == 'all':  # combine label's shape，dimension 2 to dimension 1
            temp_label = np.unique(label[i])
            if temp_label.size == 1:
                index[i] = True
                label_new.append(label[i, 0])
        elif mode == 'any':
            index[i] = True
            if np.any(label[i] > 0):
                temp_label = np.unique(label[i])
                if temp_label.size == 1:
                    label_new.append(temp_label[0])
                else:
                    label_new.append(temp_label[1])
            else:
                label_new.append(0)
        else:
            index[i] = ~index[i]
            label_new.append(label[i, 0])
    return data[index], np.array(label_new)


def reshape_data(data, merge):
    if merge == 0:
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    else:
        return data.reshape(data.shape[0] * data.shape[1] // merge, merge, data.shape[2])


def reshape_label(label, merge):
    if merge == 0:
        return label.reshape(label.shape[0] * label.shape[1])
    else:
        return label.reshape(label.shape[0] * label.shape[1] // merge, merge)


def shuffle_data_label(data, label):
    index = np.arange(data.shape[0])
    np.random.shuffle(index)
    return data[index, ...], label[index, ...]


def prepare_pretrain_dataset(data, labels, training_rate, seed=None):
    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test = partition_and_reshape(data, labels, label_index=0
                                                                                                  , training_rate=training_rate, vali_rate=0.1
                                                                                                  , change_shape=False)
    return data_train, label_train, data_vali, label_vali


def prepare_classifier_dataset(data, labels, label_index=0, training_rate=0.8, label_rate=1.0, change_shape=True
                               , merge=0, merge_mode='all', seed=None, balance=False, vali_rate = 0.1):

    set_seeds(seed)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = partition_and_reshape(data, labels, label_index=label_index, training_rate=training_rate, vali_rate=vali_rate
                                , change_shape=change_shape, merge=merge, merge_mode=merge_mode)
    set_seeds(seed)
    if balance:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset_balance(data_train, label_train, training_rate=label_rate)
    elif training_rate>0:
        data_train_label, label_train_label, _, _ \
            = prepare_simple_dataset(data_train, label_train, training_rate=label_rate)
    else:
        data_train_label = None
        label_train_label = None
    return data_train_label, label_train_label, data_vali, label_vali, data_test, label_test


def partition_and_reshape(data, labels, label_index=0, training_rate=0.8, vali_rate=0.1, change_shape=True
                          , merge=0, merge_mode='all', shuffle=True):
    arr = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    vali_num = int(data.shape[0] * vali_rate)
    data_train = data[:train_num, ...]
    data_vali = data[train_num:train_num+vali_num, ...]
    data_test = data[train_num+vali_num:, ...]
    t = np.min(labels[:, :, label_index])
    label_train = labels[:train_num, ..., label_index] - t
    label_vali = labels[train_num:train_num+vali_num, ..., label_index] - t
    label_test = labels[train_num+vali_num:, ..., label_index] - t
    if change_shape:
        data_train = reshape_data(data_train, merge)
        data_vali = reshape_data(data_vali, merge)
        data_test = reshape_data(data_test, merge)
        label_train = reshape_label(label_train, merge)
        label_vali = reshape_label(label_vali, merge)
        label_test = reshape_label(label_test, merge)
    data_train, label_train = merge_dataset(data_train, label_train, mode=merge_mode)
    data_test, label_test = merge_dataset(data_test, label_test, mode=merge_mode)
    data_vali, label_vali = merge_dataset(data_vali, label_vali, mode=merge_mode)
    print('Train Size: %d, Vali Size: %d, Test Size: %d' % (label_train.shape[0], label_vali.shape[0], label_test.shape[0]))
    return data_train, label_train, data_vali, label_vali, data_test, label_test


def prepare_simple_dataset(data, labels, training_rate=0.2):
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
    data = data[arr]
    labels = labels[arr]
    train_num = int(data.shape[0] * training_rate)
    data_train = data[:train_num, ...]
    data_test = data[train_num:, ...]
    t = np.min(labels)
    label_train = labels[:train_num] - t
    label_test = labels[train_num:] - t
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    print('Label Size: %d, Unlabel Size: %d. Label Distribution: %s'
          % (label_train.shape[0], label_test.shape[0], ', '.join(str(e) for e in label_num)))
    return data_train, label_train, data_test, label_test


def prepare_simple_dataset_balance(data, labels, training_rate=0.8):
    labels_unique = np.unique(labels)
    label_num = []
    for i in range(labels_unique.size):
        label_num.append(np.sum(labels == labels_unique[i]))
    train_num = min(min(label_num), int(data.shape[0] * training_rate / len(label_num)))
    if train_num == min(label_num):
        print("Warning! You are using all of label %d." % label_num.index(train_num))
    index = np.zeros(data.shape[0], dtype=bool)
    for i in range(labels_unique.size):
        class_index = np.argwhere(labels == labels_unique[i])
        class_index = class_index.reshape(class_index.size)
        np.random.shuffle(class_index)
        temp = class_index[:train_num]
        index[temp] = True
    t = np.min(labels)
    data_train = data[index, ...]
    data_test = data[~index, ...]
    label_train = labels[index, ...] - t
    label_test = labels[~index, ...] - t
    print('Balance Label Size: %d, Unlabel Size: %d; Real Label Rate: %0.3f' % (label_train.shape[0], label_test.shape[0]
                                                               , label_train.shape[0] * 1.0 / labels.size))
    return data_train, label_train, data_test, label_test


def regularization_loss(model, lambda1, lambda2):
    l1_regularization = 0.0
    l2_regularization = 0.0
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)
        l2_regularization += torch.norm(param, 2)
    return lambda1 * l1_regularization, lambda2 * l2_regularization


def match_labels(labels, labels_targets):
    index = np.zeros(labels.size, dtype=np.bool)
    for i in range(labels_targets.size):
        index = index | (labels == labels_targets[i])
    return index


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Normalization(Pipeline):
    """ Pre-processing steps for pretraining"""
    def __init__(self, feature_len, norm_acc=True, norm_mag=True, gamma=1.0):
        super().__init__()
        self.feature_len = feature_len
        self.norm_acc = norm_acc
        self.norm_mag = norm_mag
        self.eps = 1e-5
        self.acc_norm = 9.8
        self.gamma = gamma
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

    def __call__(self, instance):
        # masked normalization
        instance_new = instance.copy()[:, :self.feature_len]
        if instance_new.shape[1] >= 6 and self.norm_acc:
            instance_new[:, :3] = instance_new[:, :3] / self.acc_norm
        if instance_new.shape[1] == 9 and self.norm_mag:
            mag_norms = np.linalg.norm(instance_new[:, 6:9], axis=1) + self.eps
            mag_norms = np.repeat(mag_norms.reshape(mag_norms.size, 1), 3, axis=1)
            instance_new[:, 6:9] = instance_new[:, 6:9] / mag_norms * self.gamma
        return instance_new


class Preprocess4Mask:
    """ Pre-processing steps for pretraining"""
    def __init__(self, mask_cfg):
        self.mask_ratio = mask_cfg.mask_ratio  # masking probability
        self.mask_alpha = mask_cfg.mask_alpha
        self.max_gram = mask_cfg.max_gram
        self.mask_prob = mask_cfg.mask_prob
        self.replace_prob = mask_cfg.replace_prob

    def gather(self, data, position1, position2):
        result = []
        for i in range(position1.shape[0]):
            result.append(data[position1[i], position2[i]])
        return np.array(result)

    def mask(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.zeros(position2[i].size)
        return data

    def replace(self, data, position1, position2):
        for i in range(position1.shape[0]):
            data[position1[i], position2[i]] = np.random.random(position2[i].size)
        return data

    def __call__(self, instance):
        shape = instance.shape

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = max(1, int(round(shape[0] * self.mask_ratio)))

        # For masked Language Models
        # mask_pos = bert_mask(shape[0], n_pred)
        mask_pos = span_mask(shape[0], self.max_gram,  goal_num_predict=n_pred)

        instance_mask = instance.copy()

        if isinstance(mask_pos, tuple):
            mask_pos_index = mask_pos[0]
            if np.random.rand() < self.mask_prob:
                self.mask(instance_mask, mask_pos[0], mask_pos[1])
            elif np.random.rand() < self.replace_prob:
                self.replace(instance_mask, mask_pos[0], mask_pos[1])
        else:
            mask_pos_index = mask_pos
            if np.random.rand() < self.mask_prob:
                instance_mask[mask_pos, :] = np.zeros((len(mask_pos), shape[1]))
            elif np.random.rand() < self.replace_prob:
                instance_mask[mask_pos, :] = np.random.random((len(mask_pos), shape[1]))
        seq = instance[mask_pos_index, :]
        return instance_mask, np.array(mask_pos_index), np.array(seq)


class IMUDataset(Dataset):
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, labels, pipeline=[], isNormalization=True):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.labels = labels
        self.feature_len = 6
        # instance norm,when generate embedding using instance norm，when fine-tune not use
        if isNormalization:
            self.instance_norm = nn.InstanceNorm1d(self.feature_len)
            self.data = torch.tensor(self.data.transpose((0,2,1)))
            self.data = self.instance_norm(self.data)
            self.data = self.data.numpy().transpose((0,2,1))

    def __getitem__(self, index):
        instance = self.data[index]
        for proc in self.pipeline:
            instance = proc(instance)
        return torch.from_numpy(instance).float(), torch.from_numpy(np.array(self.labels[index])).long()

    def __len__(self):
        return len(self.data)


class LIBERTDataset4Pretrain(Dataset): 
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, data, pipeline=[]):  
        self.pipeline = pipeline
        self.data = data
        self.feature_len = 6
        self.instance_norm = nn.InstanceNorm1d(self.feature_len)

        self.aug1, self.aug2 = DataTransform(self.data)  # self.aug1:weak augmentation,self.aug2:strong augmentation

        # instance normalization
        mean = np.mean(self.aug1, axis=1)
        var = np.var(self.aug1, axis=1)
        self.aug1 = torch.tensor(self.aug1.transpose((0,2,1)))
        self.aug1 = self.instance_norm(self.aug1)
        self.aug1 = self.aug1.numpy().transpose((0,2,1))
        mean = np.mean(self.aug1, axis=1)
        var = np.var(self.aug1, axis=1)
        self.aug2 = torch.tensor(self.aug2.transpose((0,2,1)))
        self.aug2 = self.instance_norm(self.aug2)
        self.aug2 = self.aug2.numpy().transpose((0,2,1))

    def __getitem__(self, index):
        instance_1 = self.aug1[index]
        instance_2 = self.aug2[index]
        for proc in self.pipeline:
            instance_1 = proc(instance_1)  
            instance_2 = proc(instance_2) 
        mask_seq_1, masked_pos_1, seq_1 = instance_1
        mask_seq_2, masked_pos_2, seq_2 = instance_2
        return torch.from_numpy(mask_seq_1), torch.from_numpy(masked_pos_1).long(), torch.from_numpy(seq_1), \
            torch.from_numpy(mask_seq_2), torch.from_numpy(masked_pos_2).long(), torch.from_numpy(seq_2)

    def __len__(self):
        return len(self.data)


def handle_argv(target, config_train, prefix):
    parser = argparse.ArgumentParser(description='PyTorch CrossHAR Model')
    parser.add_argument('-mv', '--model_version', type=str, default='v1', help='Model config')
    parser.add_argument('-d', '--dataset', type=str, default='uci', help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib'])
    parser.add_argument('-td', '--target_dataset', type=str, default='uci', help='Dataset name', choices=['hhar', 'motion', 'uci', 'shoaib'])
    parser.add_argument('-dv', '--dataset_version',  type=str, default='20_120', help='Dataset version', choices=['20_120'])
    parser.add_argument('-g', '--gpu', type=str, default=None, help='Set specific GPU')
    parser.add_argument('-f', '--model_file', type=str, default='', help='Pretrain model file')
    parser.add_argument('-t', '--train_cfg', type=str, default='./config/' + config_train, help='Training config json file path')
    parser.add_argument('-a', '--mask_cfg', type=str, default='./config/mask.json', help='Mask strategy json file path')
    parser.add_argument('-l', '--label_index', type=int, default=0, help='Label Index')
    parser.add_argument('-s', '--save_model', type=str, default='model', help='The saved model name')
    parser.add_argument('-lr', '--label_rate', type=float, default=0.1, help='use finetune data ratio')
    parser.add_argument('-am', '--augument_method', type=str, default='channel_aug')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    model_cfg = load_model_config(target, prefix, args.model_version)
    if model_cfg is None:
        print("Unable to find corresponding model config!")
        sys.exit()
    args.model_cfg = model_cfg
    dataset_cfg = load_dataset_stats(args.dataset, args.dataset_version)
    if dataset_cfg is None:
        print("Unable to find corresponding dataset config!")
        sys.exit()
    args.dataset_cfg = dataset_cfg
    args = create_io_config(args, args.dataset, args.dataset_version, pretrain_model=args.model_file, target=target)
    return args


def load_raw_data(args):
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels


def load_pretrain_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    mask_cfg = MaskConfig.from_json(args.mask_cfg)
    dataset_cfg = args.dataset_cfg
    if model_cfg.feature_num > dataset_cfg.dimension:
        print("Bad Crossnum in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    print('load pretrain data from:'+args.data_path)
    data = np.load(args.data_path).astype(np.float32)
    print('load pretrain label from:'+args.label_path)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg


def load_classifier_data_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_cfg, dataset_cfg


def load_classifier_config(args):
    model_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    set_seeds(train_cfg.seed)
    return train_cfg, model_cfg, dataset_cfg


def load_bert_classifier_data_config(args):
    model_bert_cfg, model_classifier_cfg = args.model_cfg
    train_cfg = TrainConfig.from_json(args.train_cfg)
    dataset_cfg = args.dataset_cfg
    if model_bert_cfg.feature_num > dataset_cfg.dimension:
        print("Bad feature_num in model cfg")
        sys.exit()
    set_seeds(train_cfg.seed)
    data = np.load(args.data_path).astype(np.float32)
    labels = np.load(args.label_path).astype(np.float32)
    return data, labels, train_cfg, model_bert_cfg, model_classifier_cfg, dataset_cfg


def augument_dataset(data, label, method='channel_aug'):
    # data(sample num, sequence_len, feature)
    # label(sample num, sequence_len, feature)
    print(f'begin data augmentation: method is {method}')
    data_res = np.empty((0,data.shape[1],data.shape[2]))
    label_res = np.empty((0,label.shape[1],label.shape[2]))
    aug_num = 5 # aug_num: the number of sample of one sample will augument

    if method == 'channel_aug':
        lst = [0,1,2]
        permutations_list = list(permutations(lst))
        for perm in permutations_list:
            perm = list(perm)
            perm2 = [i+3 for i in perm]
            data_temp = data[:, :, perm+perm2]
            data_res = np.concatenate([data_res, data_temp], axis=0)
            label_res = np.concatenate([label_res, label], axis=0)
    
    elif method == 'rotation_random':
        axis_num = 3
        # each sample rotation matrix is same
        for i in range(aug_num):
            axis = np.random.uniform(low=-1, high=1, size=axis_num)
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            # rotation_mat = axangle2mat(axis,angle)
            rotation_mat = special_ortho_group.rvs(3)
            data_temp = data.reshape(-1,6)
            aug_temp_acc = np.matmul(data_temp[:,:3], rotation_mat)
            aug_temp_gyr = np.matmul(data_temp[:,3:], rotation_mat)
            aug_temp = np.concatenate([aug_temp_acc, aug_temp_gyr], axis=1)
            aug_temp = aug_temp.reshape(-1,120,6)
            data_res = np.concatenate([data_res,aug_temp],axis=0)
            label_res = np.concatenate([label_res,label],axis=0)
        data_res = np.concatenate([data_res,data],axis=0)
        label_res = np.concatenate([label_res,label],axis=0)
    
    elif method == 'jitter':
        sigma = 0.05
        for j in range(aug_num):
            myNoise = np.random.normal(loc=0, scale=sigma, size=data.shape)
            data_temp = data + myNoise
            data_res = np.concatenate([data_res,data_temp], axis=0)
            label_res = np.concatenate([label_res,label], axis=0)
        data_res = np.concatenate([data_res,data],axis=0)
        label_res = np.concatenate([label_res,label],axis=0)

    elif method == 'scaling':
        # return X*myNoise
        axis_num = 3
        for i in range(aug_num):
            scalingFactor = np.random.normal(loc=1.0, scale=0.1, size=(1,axis_num))
            myNoise = np.matmul(np.ones((data.shape[1],1)), scalingFactor)
            for j in range(data.shape[0]):
                data_temp = data[j]
                label_temp = label[j]                
                aug_temp_acc = data_temp[:,:3]*myNoise
                aug_temp_gyr = data_temp[:,3:]*myNoise
                aug_temp = np.concatenate([aug_temp_acc, aug_temp_gyr], axis=1)
                data_res = np.concatenate([data_res,np.expand_dims(aug_temp,axis=0)],axis=0)
                label_res = np.concatenate([label_res,np.expand_dims(label_temp,axis=0)],axis=0)
        data_res = np.concatenate([data_res,data],axis=0)
        label_res = np.concatenate([label_res,label],axis=0)

    elif method == 'permutation':
        nPerm = 4
        minSegLength = 10
        for i in range(aug_num):
            X_new = np.zeros(data.shape)
            idx = np.random.permutation(nPerm)
            bWhile = True
            while bWhile == True:
                segs = np.zeros(nPerm+1, dtype=int)
                segs[1:-1] = np.sort(np.random.randint(minSegLength, data.shape[1]-minSegLength, nPerm-1))
                segs[-1] = data.shape[1]
                if np.min(segs[1:]-segs[0:-1]) > minSegLength:
                    bWhile = False
            
            pp = 0
            for ii in range(nPerm):
                x_temp = data[:, segs[idx[ii]]:segs[idx[ii]+1], :]
                X_new[:, pp:pp+x_temp.shape[1],:] = x_temp
                pp += x_temp.shape[1]
            data_res = np.concatenate([data_res,X_new],axis=0)
            label_res = np.concatenate([label_res,label],axis=0)
        data_res = np.concatenate([data_res,data],axis=0)
        label_res = np.concatenate([label_res,label],axis=0)

    else:
        print('method not exist')
        return None, None
        
    print(f'data augumentation end')
    return data_res, label_res


if __name__ == '__main__':
    pass


