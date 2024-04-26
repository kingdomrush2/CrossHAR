import numpy as np
import torch
import torch.nn as nn
import models, train

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchinfo import summary
from models import LIMUBertModel4Pretrain
from utils import get_device, LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask, augument_dataset
from TC import TC


def main(args, training_rate):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

    data, labels = augument_dataset(data, labels, method=args.augument_method)  # dataset augumentation

    pipeline = [Preprocess4Mask(mask_cfg)]
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)
    print('data_train size:', len(data_train))
    print('data_val size:', len(data_test))

    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    print('pretrain batch_size:', train_cfg.batch_size)
    print('pretrain epoch:', train_cfg.n_epochs)
    print('learning rate:', train_cfg.lr)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size, drop_last=True)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size, drop_last=True)

    device = get_device(args.gpu)
    LIMUBert_model = LIMUBertModel4Pretrain(model_cfg).to(device)
    summary(LIMUBert_model, (1, data.shape[1], data.shape[2]))
    TC_model = TC().to(device)
    summary(TC_model, (1, data.shape[1], 72))

    criterion = nn.MSELoss(reduction='none')

    LIMUBert_optimizer = torch.optim.Adam(params=LIMUBert_model.parameters(), lr=train_cfg.lr)
    TC_optimizer = torch.optim.Adam(params=TC_model.parameters(), lr=train_cfg.lr)
    print('pretrain model save path:'+args.save_path_pretrain)
    trainer = train.Trainer(train_cfg, LIMUBert_model, LIMUBert_optimizer, TC_model, TC_optimizer, args.save_path_pretrain,
                            device, batch_size=train_cfg.batch_size, criterion=criterion)

    print('dataloader_train',len(data_loader_train))
    print('dataloader_test',len(data_loader_test))
    trainer.pretrain(data_loader_train, data_loader_test)


if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    main(args, training_rate)
