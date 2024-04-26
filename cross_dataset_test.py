import torch
import numpy as np
import train

from torch import nn
from torch.utils.data import DataLoader
from config import load_dataset_label_names
from models import LIMUBertModel4Pretrain, fetch_classifier
from utils import LIBERTDataset4Pretrain, load_pretrain_data_config, get_device, handle_argv, \
    Preprocess4Normalization, IMUDataset, prepare_classifier_dataset, load_classifier_config
from TC import TC
from finetune_train import FinetuneTrainer

'''
This program conducts cross-dataset testing, using the model trained on the source dataset
to generate embeddings for the target dataset. 
It then tests the accuracy of the target dataset on the linear classifier of the source dataset.
'''

if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    source_dataset = args.dataset
    target_dataset = args.target_dataset
    print('source dataset:', source_dataset)
    print('target dataset:', target_dataset)
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

    # Reload the data of the target dataset.
    data_path = './dataset/' + target_dataset + '/data_20_120.npy'
    print('load target dataset data from:'+data_path)
    data = np.load(data_path).astype(np.float32)
    labels_path = './dataset/' + target_dataset + '/label_20_120.npy'
    print('load target dataset label from:'+labels_path)
    labels = np.load(labels_path).astype(np.float32)

    data_set = IMUDataset(data, labels, pipeline=[])
    data_loader = DataLoader(data_set, shuffle=False, batch_size=train_cfg.batch_size)
    LIMUBert_model = LIMUBertModel4Pretrain(model_cfg, output_embed=True)
    TC_model = TC()
    criterion = nn.MSELoss(reduction='none')

    optimizer = None
    trainer = train.Trainer(train_cfg, LIMUBert_model, optimizer, TC_model, optimizer, args.save_path_pretrain,
                            get_device(args.gpu), train_cfg.batch_size, criterion)
    target_embedding = trainer.output_embedding(data_loader, args.save_path_pretrain)

    label_index = args.label_index
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test \
        = prepare_classifier_dataset(target_embedding, labels, label_index=label_index, training_rate=0.0
                                     , label_rate=1.0, merge=model_cfg.seq_len, seed=train_cfg.seed
                                     , balance=False, change_shape=False,vali_rate=0.0)  # label_index0, merge20
    data_set_test = IMUDataset(data_test, label_test, isNormalization=False)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

    method = "transformer"
    print(method)
    criterion = nn.CrossEntropyLoss()
    args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
    _, model_cfg, dataset_cfg = load_classifier_config(args)
    model = fetch_classifier(method, model_cfg, input=data_test.shape[-1], output=label_num)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)  
    trainer = FinetuneTrainer(train_cfg, model, optimizer, criterion, get_device(args.gpu), train_cfg.batch_size,
                              args.save_path_classifier, label_num, method=method, source_dataset=args.dataset, 
                              target_dataset=args.target_dataset)
    conf_matrix, acc, precision, recall, f1_macro, f1_micro = trainer.test(data_loader_test, len(data_test))
    print('acc:{:.4f},precision:{:.4f},recall:{:.4f},f1_macor:{:.4f},f1_micor:{:.4f}'
              .format(acc,precision,recall,f1_macro,f1_micro))
    torch.set_printoptions(sci_mode=False)
    print('conf_matrix')
    print(conf_matrix)


    