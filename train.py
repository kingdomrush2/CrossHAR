import time

import numpy as np
import torch
import pandas as pd

from loss import NTXentLoss


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, masked_model, masked_optimizer, Contrastive_model, Contrastive_optimizer, save_path, device, batch_size, criterion):
        self.cfg = cfg # config for training : see class Config
        self.masked_model = masked_model
        self.masked_optimizer = masked_optimizer
        self.Contrastive_model = Contrastive_model
        self.Contrastive_optimizer = Contrastive_optimizer
        self.save_path = save_path
        self.device = device  # device name
        self.batch_size = batch_size
        self.criterion = criterion
        self.lambda1 = 6
        self.lambda2 = 1
        self.beizhu = ''

    def pretrain(self, data_loader_train, data_loader_test, model_file=None):
        print('mlm loss: ct loss='+str(self.lambda1)+':'+str(self.lambda2))
        n_epoch_now = 0  # define epoch that model begin to train

        self.load(model_file)
        self.masked_model = self.masked_model.to(self.device)
        self.Contrastive_model = self.Contrastive_model.to(self.device)

        self.masked_model.train()
        self.Contrastive_model.train()

        global_step = 0 # global iteration steps regardless of epochs
        best_loss = 1e6

        for e in range(n_epoch_now, self.cfg.n_epochs):
            loss_sum, loss_lm_sum, loss_nt_sum = 0.0, 0.0, 0.0 # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            for i, (mask_seqs_1, masked_pos_1, seqs_1, mask_seqs_2, masked_pos_2, seqs_2) in enumerate(data_loader_train):
                start_time = time.time()
                mask_seqs_1, masked_pos_1, seqs_1 = mask_seqs_1.to(self.device), masked_pos_1.to(self.device), seqs_1.to(self.device)
                mask_seqs_2, masked_pos_2, seqs_2 = mask_seqs_2.to(self.device), masked_pos_2.to(self.device), seqs_2.to(self.device)
                self.masked_optimizer.zero_grad()
                self.Contrastive_optimizer.zero_grad()

                representation_1, seq_recon_1 = self.masked_model(mask_seqs_1, masked_pos_1)
                loss_lm_1 = self.criterion(seq_recon_1, seqs_1)
                loss_lm_1 = loss_lm_1.mean()  # mean() for Data Parallelism
                
                representation_2, seq_recon_2 = self.masked_model(mask_seqs_2, masked_pos_2)  # shape(128,120time_step,72feature_num)
                loss_lm_2 = self.criterion(seq_recon_2, seqs_2)
                loss_lm_2 = loss_lm_2.mean()  # mean() for Data Parallelism

                zis = self.Contrastive_model(representation_1)
                zjs = self.Contrastive_model(representation_2)

                nt_xent_criterion = NTXentLoss(device=self.device, batch_size=self.batch_size)
                loss_nt = nt_xent_criterion(zis, zjs)

                #reset best_loss
                if e==(self.cfg.n_epochs-self.cfg.n_epochs_cl):
                    best_loss = 10e6

                if e<(self.cfg.n_epochs-self.cfg.n_epochs_cl):
                    loss = (loss_lm_1 + loss_lm_2) / 2  # only use mlm loss
                    # loss = loss_nt
                    # loss = self.lambda1*(loss_lm_1 + loss_lm_2)/2 + self.lambda2*loss_nt
                else:
                    loss = self.lambda1*(loss_lm_1 + loss_lm_2)/2 + self.lambda2*loss_nt
                    # loss = (loss_lm_1 + loss_lm_2) / 2
                    # loss = loss_nt

                # loss = self.lambda1*(loss_lm_1 + loss_lm_2)/2 + self.lambda2*loss_nt
                # loss = (loss_lm_1 + loss_lm_2) / 2  
                # loss = loss_nt  # only use contrastive loss

                loss.backward()
                self.masked_optimizer.step()
                self.Contrastive_optimizer.step()

                time_sum += time.time() - start_time
                global_step += 1
                loss_sum += loss.item()
                loss_lm_sum += ((loss_lm_1 + loss_lm_2)/2).item()
                loss_nt_sum += loss_nt.item()

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return

            loss_eva, loss_eva_mlm, loss_eva_nt = self.run(data_loader_test, e)
            dlt_len = len(data_loader_train)
            print('Epoch %d/%d : Train Loss %5.4f. lm loss %5.4f nt loss %5.4f | Val Loss %5.4f lm loss %5.4f nt loss %5.4f'
                  % (e + 1, self.cfg.n_epochs, loss_sum / dlt_len, loss_lm_sum/dlt_len, loss_nt_sum/dlt_len, loss_eva,
                     loss_eva_mlm, loss_eva_nt))
            if loss_eva < best_loss:
                best_loss = loss_eva
                self.save()
        print('The Total Epoch have been reached.')


    def run(self, data_loader, e):
        """ Evaluation Loop """
        self.masked_model.eval() # evaluation mode
        self.Contrastive_model.eval()
        results = [] # prediction results
        labels = []
        time_sum = 0.0
        loss_sum, loss_mlm_sum, loss_nt_sum = 0.0, 0.0, 0.0
        for mask_seqs_1, masked_pos_1, seqs_1, mask_seqs_2, masked_pos_2, seqs_2 in data_loader:
            mask_seqs_1, masked_pos_1, seqs_1 = mask_seqs_1.to(self.device), masked_pos_1.to(self.device), seqs_1.to(
                self.device)
            mask_seqs_2, masked_pos_2, seqs_2 = mask_seqs_2.to(self.device), masked_pos_2.to(self.device), seqs_2.to(
                self.device)
            with torch.no_grad():  # evaluation without gradient calculation
                representation_1, seq_recon_1 = self.masked_model(mask_seqs_1, masked_pos_1)
                loss_lm_1 = self.criterion(seq_recon_1, seqs_1)
                loss_lm_1 = loss_lm_1.mean()  # mean() for Data Parallelism
                
                representation_2, seq_recon_2 = self.masked_model(mask_seqs_2, masked_pos_2)  # shape(128,120time_step,72feature_num)
                loss_lm_2 = self.criterion(seq_recon_2, seqs_2)
                loss_lm_2 = loss_lm_2.mean()  # mean() for Data Parallelism

                zis = self.Contrastive_model(representation_1)
                zjs = self.Contrastive_model(representation_2)

                nt_xent_criterion = NTXentLoss(device=self.device, batch_size=self.batch_size)
                loss_nt = nt_xent_criterion(zis, zjs)

                # if e<(self.cfg.n_epochs-self.cfg.n_epochs_cl):
                #     loss = (loss_lm_1 + loss_lm_2) / 2  
                # else:
                #     loss = self.lambda1*(loss_lm_1 + loss_lm_2)/2 + self.lambda2*loss_nt

                if e<(self.cfg.n_epochs-self.cfg.n_epochs_cl):
                    loss = (loss_lm_1 + loss_lm_2) / 2 
                    # loss = loss_nt
                    # loss = self.lambda1*(loss_lm_1 + loss_lm_2)/2 + self.lambda2*loss_nt
                else:
                    loss = self.lambda1*(loss_lm_1 + loss_lm_2)/2 + self.lambda2*loss_nt
                    # loss = (loss_lm_1 + loss_lm_2) / 2
                    # loss = loss_nt

                # loss = self.lambda1*(loss_lm_1 + loss_lm_2)/2 + self.lambda2*loss_nt
                # loss = (loss_lm_1 + loss_lm_2) / 2  
                # loss = loss_nt  
                loss_mlm_sum += ((loss_lm_1 + loss_lm_2) / 2).item()
                loss_nt_sum += loss_nt.item()
                loss_sum += loss.item()
        return loss_sum/len(data_loader), loss_mlm_sum/len(data_loader), loss_nt_sum/len(data_loader)  
        # calculate each batch's average loss

    def output_embedding(self, data_loader, model_file=None, data_parallel=False, load_self=False):
        """ Evaluation Loop """
        self.masked_model.eval() # evaluation mode
        self.Contrastive_model.eval()
        self.load(model_file, load_self=load_self)
        self.masked_model = self.masked_model.to(self.device)
        self.Contrastive_model = self.Contrastive_model.to(self.device)
        results = []
        for seqs, label in data_loader:
            seqs, label = seqs.to(self.device), label.to(self.device)
            with torch.no_grad():  # evaluation without gradient calculation
                representation = self.masked_model(seqs)
                results.append(representation)
        return torch.cat(results, 0).cpu().numpy()

    def test_inferece_time(self, data_loader, model_file=None, data_parallel=False, load_self=False, classifier_model=None):
        """ Calculate inferece time per sample """
        self.masked_model.eval() # evaluation mode
        self.Contrastive_model.eval()
        classifier_model.eval()
        classifier_model_path = 'saved/classifier_base_transformer_uci_20_120_4activity/limu_v1_4activity_transformer.pt'
        classifier_model.load_state_dict(torch.load(classifier_model_path, map_location=self.device))
        self.load(model_file, load_self=load_self)
        self.masked_model = self.masked_model.to(self.device)
        self.Contrastive_model = self.Contrastive_model.to(self.device)
        results = []
        inferece_time_list = []
        num = 0
        for seqs, label in data_loader:
            if num == 100:
                data = np.array(inferece_time_list)
                df = pd.DataFrame(data, columns=['infere_time'])
                df.to_csv('./infere_time.csv')
                return
            num+=1
            seqs, label = seqs.to(self.device), label.to(self.device)
            with torch.no_grad():  # evaluation without gradient calculation
                start_time = time.time()
                representation = self.masked_model(seqs)
                representation = classifier_model(representation)
                predicted = torch.max(representation, 1)[1]
                end_time = time.time()
                run_time = end_time - start_time
                inferece_time_list.append(run_time)
                print("inferece time：{}second".format(run_time))
                results.append(representation)
        return torch.cat(results, 0).cpu().numpy()
    

    def load(self, model_file, load_self=False):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            if load_self:
                self.masked_model.load_self(model_file + '.pt', map_location=self.device)
            else:
                masked_model_path = model_file + '_masked_'+str(self.lambda1)+'_'+str(self.lambda2)+self.beizhu+'.pt'
                Contrastive_model_path = model_file + '_Contrastive_'+str(self.lambda1)+'_'+str(self.lambda2)+self.beizhu+'.pt'
                print('Loading the pretrain masked model from', masked_model_path)
                print('Loading the pretrain Contrastive model from', Contrastive_model_path)
                self.masked_model.load_state_dict(torch.load(masked_model_path, map_location=self.device))
                self.Contrastive_model.load_state_dict(torch.load(Contrastive_model_path, map_location=self.device))

    def save(self):
        """ save current model """
        torch.save(self.masked_model.state_dict(),  self.save_path + '_masked_'+str(self.lambda1)+'_'+str(self.lambda2)+self.beizhu+'.pt')
        torch.save(self.Contrastive_model.state_dict(), self.save_path + '_Contrastive_'+str(self.lambda1)+'_'+str(self.lambda2)+self.beizhu+'.pt')

