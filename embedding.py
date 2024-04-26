import os
import train
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from models import LIMUBertModel4Pretrain
from utils import load_pretrain_data_config, get_device, handle_argv, IMUDataset, augument_dataset
from TC import TC


def fetch_setup(args, output_embed):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

    data, labels = augument_dataset(data, labels, method=args.augument_method) 

    pipeline = []
    data_set = IMUDataset(data, labels, pipeline=pipeline)
    data_loader = DataLoader(data_set, shuffle=False, batch_size=1)

    LIMUBert_model = LIMUBertModel4Pretrain(model_cfg, output_embed=output_embed)
    TC_model = TC()
    criterion = nn.MSELoss(reduction='none')
    return data, labels, data_loader, LIMUBert_model, TC_model, criterion, train_cfg


def generate_embedding_or_output(args, save=False, output_embed=True):
    data, labels, data_loader, LIMUBert_model, TC_model, criterion, train_cfg \
        = fetch_setup(args, output_embed)

    optimizer = None
    criterion = nn.MSELoss(reduction='none')
    trainer = train.Trainer(train_cfg, LIMUBert_model, optimizer, TC_model, optimizer, args.save_path_pretrain,
                            get_device(args.gpu), train_cfg.batch_size, criterion)

    output = trainer.output_embedding(data_loader, args.save_path_pretrain)

    if save:
        save_name = 'embed' + args.model_file.split('.')[0] + '_' + args.dataset + '_' + args.dataset_version
        print('embedding save at '+'embed/'+ save_name + '.npy')
        np.save(os.path.join('embed', save_name + '.npy'), output)
        label_save_name = 'label' + args.model_file.split('.')[0] + '_' + args.dataset + '_' + args.dataset_version
        print('label save at '+'embed/'+ label_save_name + '.npy')
        np.save(os.path.join('embed', label_save_name + '.npy'), labels)
    return data, output, labels


def load_embedding_label(model_file, dataset, dataset_version):
    embed_name = 'embed' + model_file + '_' + dataset + '_' + dataset_version 
    label_name = 'label' + model_file + '_' + dataset + '_' + dataset_version
    embed_path = os.path.join('embed', embed_name + '.npy')
    print('load embedding from:'+embed_path)
    embed = np.load(embed_path).astype(np.float32)
    labels_path = os.path.join('embed', label_name + '.npy')
    print('load embedding label from:'+labels_path)
    labels = np.load(labels_path).astype(np.float32)
    return embed, labels


if __name__ == "__main__":
    save = True
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    data, output, labels = generate_embedding_or_output(args=args, output_embed=True, save=save)
