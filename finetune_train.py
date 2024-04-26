import torch
import copy
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class FinetuneTrainer(object):
    def __init__(self, train_cfg, model, optimizer, criterion, device, batch_size, save_path, label_num, method, 
                 source_dataset='', target_dataset=''):
        self.train_cfg = train_cfg
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.save_path = save_path
        self.criterion = criterion
        self.label_num = label_num
        self.method = method
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def train(self, train_loader, valid_loader, test_loader, test_sample_num):
        model = self.model.to(self.device)
        best_loss = 1e6
        best_model = model.state_dict()
        for epoch in range(self.train_cfg.n_epochs):
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                model.train()
                output = model(x)
                loss = self.criterion(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            loss = self.evaluate(model, valid_loader)
            print('epoch, loss:', epoch, loss)
            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, self.save_path+'_'+self.method+'.pt')
        conf_matrix, acc, precision, recall, f1_macro, f1_micro = self.test(test_loader, test_sample_num)
        print('acc:{:.4f},precision:{:.4f},recall:{:.4f},f1_macor:{:.4f},f1_micor:{:.4f}'
              .format(acc,precision,recall,f1_macro,f1_micro))
        torch.set_printoptions(sci_mode=False)
        print('conf_matrix')
        print(conf_matrix)

    def evaluate(self, model, val_iter):
        model.eval()
        loss = 0.0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_iter):
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                loss += self.criterion(output, y)
        return loss / len(val_iter)

    def test(self, test_iter, test_sample_num):
        print('load classifier from:'+self.save_path+'_'+self.method+'.pt')
        self.model.load_state_dict(torch.load(self.save_path+'_'+self.method+'.pt'))
        self.model.eval()
        self.model = self.model.to(self.device)
        y_true, y_pred = [], []
        with torch.no_grad():
            conf_matrix = torch.zeros(self.label_num, self.label_num)
            correct = 0
            print(len(test_iter))
            for i, (x, y) in enumerate(test_iter):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                predicted = torch.max(output, 1)[1]

                y_true = y_true + y.tolist()
                y_pred = y_pred + predicted.tolist()

                correct += (predicted == y).sum()
        conf_matrix = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        class_list = ['W', 'U', 'D', 'S']
        return conf_matrix, acc, precision, recall, f1_macro, f1_micro