import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from config import  load_dataset_label_names
from embedding import load_embedding_label
from models import fetch_classifier
from torchinfo import summary
from utils import get_device, handle_argv, IMUDataset, load_classifier_config, prepare_classifier_dataset
from finetune_train import FinetuneTrainer


def classify_embeddings(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None):
    train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = prepare_classifier_dataset(data, labels, label_index=label_index, training_rate=training_rate
                                     , label_rate=label_rate, merge=model_cfg.seq_len, seed=train_cfg.seed
                                     , balance=balance, change_shape=False)  # label_index0, merge20
    data_set_train = IMUDataset(data_train, label_train, isNormalization=False)
    data_set_vali = IMUDataset(data_vali, label_vali, isNormalization=False)
    data_set_test = IMUDataset(data_test, label_test, isNormalization=False)
    print('fine-tune batch_size:', train_cfg.batch_size)
    print('fine-tune epoch:', train_cfg.n_epochs)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    device = get_device(args.gpu)
    model = fetch_classifier(method, model_cfg, input=data_train.shape[-1], output=label_num)
    model = model.to(device)
    summary(model, (1,data.shape[1], data.shape[2]))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr) 
    trainer = FinetuneTrainer(train_cfg, model, optimizer, criterion, get_device(args.gpu), train_cfg.batch_size,
                            args.save_path_classifier, label_num, method=method, source_dataset=args.dataset, 
                            target_dataset=args.target_dataset)

    trainer.train(data_loader_train, data_loader_vali, data_loader_test, len(data_test))


if __name__ == "__main__":

    training_rate = 0.8  # unlabeled sample / total sample
    balance = False

    mode = "base"
    method = "transformer"
    print(method)
    args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
    label_rate = args.label_rate  # labeled sample / unlabeled sample
    embedding, labels = load_embedding_label(args.model_file, args.dataset, args.dataset_version)  

    classify_embeddings(args, embedding, labels, args.label_index, training_rate, label_rate, balance=balance, method=method)

