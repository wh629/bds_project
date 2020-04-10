"""
Class for learning
"""

import torch
import tqdm
import torch.nn as nn
import torch.optim as opt
import numpy as np
from sklearn import metrics
import logging as log
import math
import os
import pickle

class Learner():
    def __init__(self,
                 model,                     # model to train
                 device,                    # device to run on
                 train_data,                # training dataloader
                 val_data,                  # validation dataloader
                 test_data,                 # test dataloader
                 rating_w,                  # weight for rating loss
                 flag_w,                    # weight for flag loss
                 max_epochs,                # maximum number of epochs
                 save_path,                 # path to save best weights
                 rating_loss=None,          # loss object for rating objective
                 flag_loss=None,            # loss object for flag objective
                 optimizer=None,            # optimizer for training
                 scheduler=None,            # scheduler for optimizer
                 lr=0.001,                  # optimizer learning rate
                 weight_decay=0.01,         # optimizer weight decay
                 pct_start = 0.3,           # scheduler warm-up percentage
                 anneal_strategy='linear',  # scheduler annealing strategy
                 cycle_momentum=False,      # scheduler whether to alternate momentums
                 log_int=1e4,               # logging interval
                 buffer_break=False,        # whether to do early breaking
                 break_int=10,              # buffer before early breaking
                 f1_average='macro'         # averaging method for multi-class F1
                 ):
        """
        Class for learning
        """
        
        # set attributes
        self.model = model.to(device)
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.rating_w = rating_w
        self.flag_w = flag_w
        self.max_epochs = max_epochs
        self.save_path = save_path
        self.rating_loss = rating_loss
        self.flag_loss = flag_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_log_int = math.min(len(train_data.dataset)-1, log_int)
        self.val_log_int = math.min(len(val_data.dataset)-1, log_int)
        self.buffer_break = buffer_break
        self.break_int = break_int
        self.f1_avg = f1_average
        
        
        if rating_loss == None:
            self.rating_loss = nn.CrossEntropyLoss()
        
        if flag_loss == None:
            self.flag_loss = nn.CrossEntropyLoss()
            
        if optimizer == None:
            self.optimizer = opt.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=0.01
                )
        
        if scheduler == None:
            self.scheduler = opt.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=lr,
                epochs=max_epochs,
                steps_per_epoch=len(train_data.dataset),
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                cycle_momentum=cycle_momentum
                )
        
        # if directory for best weights does not exist, create it
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        
    def train(self, 
              epoch # integer of epoch
              ):
        """
        Training loop for a single epoch.
        
        Returns: Results as dictionary
        
        {
            "avg_loss"      : <Average Testing Loss>
            "avg_rating_acc": <Average Testing Rating Accuracy>
            "avg_flag_acc"  : <Average Testing Flag Accuracy>
            "avg_rating_f1" : <Average Testing Rating F1>
            "avg_flag_f1"   : <Average Testing Flag F1>
            }
        
        NOT TESTED YET
        """
        log.info('Training | Epoch {} out of {}'\
                 .format(epoch+1, self.max_epochs+1))
        
        cum_loss = 0
        cum_rating_acc = 0
        cum_flag_acc = 0
        cum_rating_f1 = 0
        cum_flag_f1 = 0
            
        for i, (data, label) in enumerate(tqdm(self.train_data)):        
            # put model in train mode
            self.model.train()

            # send data and labels to device
            data, labels = data.to(self.device), label.to(self.device)
            r_label = labels[:,0] # collate has rating labels 1st
            f_label = labels[:,1] # collate has flag labels 2nd
    
            # zero gradients related to optimizer
            self.optimizer.zero_grad()
    
            # send data through model forward
            r_logits, f_logits = self.model(data)
    
            # calculate loss
            l = self.rating_loss(r_logits, r_label)*self.rating_w \
                + self.flag_loss(f_logits, f_label)*self.flag_w
            # calculate gradients through back prop
            l.backward()
    
            #take a step in optimizer and scheduler
            self.optimizer.step()
            self.scheduler.step()
            
            with torch.no_grad():
                # accumulate measures
                cum_loss += l
            
                _, r_idx = torch.max(r_logits, dim=1)
                _, f_idx = torch.max(f_logits, dim=1)
                
                # prepare data for numpy metrics
                if self.device == 'cuda':
                    r_label, f_label = r_label.to('cpu'), f_label.to('cpu')
                    r_idx, f_idx = r_idx.to('cpu'), f_idx.to('cpu')
                
                cum_rating_acc += metrics.accuracy_score(r_label, r_idx)
                cum_flag_acc += metrics.accuracy_score(f_label, f_idx)
                cum_rating_f1 += metrics.f1_score(r_label, r_idx, average=self.f1_avg)
                cum_flag_f1 += metrics.f1_score(f_label, f_idx, average=self.f1_avg)
            
            
        return {
            "avg_loss"      : cum_loss/(i+1),
            "avg_rating_acc": cum_rating_acc/(i+1),
            "avg_flag_acc"  : cum_flag_acc/(i+1),
            "avg_rating_f1" : cum_rating_f1/(i+1),
            "avg_flag_f1"   : cum_flag_f1/(i+1)
            }

    def evaluate(self,
                 data_loader,  # dataloader to evaluate on
                 epoch = None, # integer of epoch number if val = True
                 val = True    # boolean whether evaluating validation data
                ):
        """
        Evaluation model on data. (May need to use dataloader as well)
    
        Returns: Results as dictionary
        
        {
            "loss"       : <Average Loss>,
            "rating_acc" : <Rating Accuracy>,
            "flag_acc"   : <Flag Accuracy>,
            "rating_f1"  : <Average Rating F1>,
            "flag_f1"    : <Average Flag F1>,
            "comb_acc"   : <Weighted Average Accuracy>,
            "comb_f1"    : <Weighted Average F1>
            }
    
        NOT TESTED YET
        """
        if val:
            log.info('Validation | Epoch {} out of {}'\
                     .format(epoch+1, self.max_epochs+1))
        
        cum_loss = 0
        cum_rating_acc = 0
        cum_flag_acc = 0
        cum_rating_f1 = 0
        cum_flag_f1 = 0
        
        # puts model in evaluation mode
        self.model.eval()
        
        # don't need to track gradient
        with torch.no_grad():
            for i, (data, labels) in enumerate(tqdm(data_loader)):
                # send data and labels to device
                data, labels = data.to(self.device), labels.to(self.device)
                r_label = labels[:,0] # collate has rating labels 1st
                f_label = labels[:,1] # collate has flag labels 2nd
                
                # send data through forward
                r_logits, f_logits = self.model(data)
                
                # calculate loss
                l = self.rating_loss(r_logits, r_label)*self.rating_w \
                    + self.flag_loss(f_logits, f_label)*self.flag_w
                
                # accumulate measures
                cum_loss += l
            
                _, r_idx = torch.max(r_logits, dim=1)
                _, f_idx = torch.max(f_logits, dim=1)
            
                # prepare data for numpy metrics
                if self.device == 'cuda':
                    r_label, f_label = r_label.to('cpu'), f_label.to('cpu')
                    r_idx, f_idx = r_idx.to('cpu'), f_idx.to('cpu')
                    
                cum_rating_acc += metrics.accuracy_score(r_label, r_idx)
                cum_flag_acc += metrics.accuracy_score(f_label, f_idx)
                cum_rating_f1 += metrics.f1_score(r_label, r_idx, average=self.f1_avg)
                cum_flag_f1 += metrics.f1_score(f_label, f_idx, average=self.f1_avg)

        return {
            "loss"       : cum_loss/(i+1),
            "rating_acc" : cum_rating_acc/(i+1),
            "flag_acc"   : cum_flag_acc/(i+1),
            "rating_f1"  : cum_rating_f1/(i+1),
            "flag_f1"    : cum_flag_f1/(i+1),
            "comb_acc"   : (self.rating_w*cum_rating_acc + self.flag_w*cum_flag_acc)/((i+1)*(self.rating_w + self.flag_w)),
            "comb_f1"    : (self.rating_w*cum_rating_f1 + self.flag_w*cum_flag_f1)/((i+1)*(self.rating_w + self.flag_w))
            }

    def learn(self,
              model_name,            # name of model used
              verbose = True,        # flag whether to print checking
              early_check = 'loss'   # key for early break check
              ):
        """
        Train model
    
        Return: Path of best weights
        
        NOT TESTED YET
        """
        best = {
            'loss'       : None,
            'rating_acc' : None,
            'flag_acc'   : None,
            'rating_f1'  : None,
            'flag_f1'    : None,
            'comb_acc'   : None,
            'comb_f1'    : None
            }
        best_path = os.path.join(self.save_path, model_name + r'_best.pt')
        best_epoch = 0
        stop = False

        for epoch in tqdm(range(self.max_epochs)):
            train_results = self.train(epoch)
            val_results = self.evaluate(self.val_data, epoch=epoch)
            
            # check for best score
            for result_type in val_results.keys():
                temp_val = val_results[result_type]
                
                if (temp_val < best[result_type] and result_type.find('loss') != -1)\
                    or temp_val > best[result_type]:
                    best[result_type] = temp_val
                    
                    if result_type == early_check:
                        torch.save(self.model.state_dict(), best_path)
                        best_epoch = epoch
                elif (
                        result_type == early_check
                        and epoch-best_epoch > self.break_int
                        and self.buffer_break
                        ):
                    stop = True
                    break
                
            if stop:
                log.info('='*40 + ' Early Stop at Epoch: {} '.format(epoch) + '='*40)
                break
    
            # print results every interval
            if epoch % self.log_int == 0 and verbose:
                log.info('Validation Information: | Stop Check Type: {} | Best Epoch: {}'\
                 ' | Best Loss: {:.4f} | Best Rating Accuracy: {:.4f} | Best Rating F1: {:.4f}'\
                     ' | Best Flag Accuracy: {:.4f} | Best Flag F1: {:.4f}'\
                         ' | Best Combined Accuracy: {:.4f} | Best Combined F1: {:.4f}'\
                             .format(
                                 early_check,
                                 best_epoch,
                                 best['loss'],
                                 best['rating_acc'],
                                 best['rating_f1'],
                                 best['flag_acc'],
                                 best['flag_f1'],
                                 best['comb_acc'],
                                 best['comb_f1']
                                 ))
        
        # log best Val information
        log.info('Validation Information: | Stop Check Type: {} | Best Epoch: {}'\
                 ' | Best Loss: {:.4f} | Best Rating Accuracy: {:.4f} | Best Rating F1: {:.4f}'\
                     ' | Best Flag Accuracy: {:.4f} | Best Flag F1: {:.4f}'\
                         ' | Best Combined Accuracy: {:.4f} | Best Combined F1: {:.4f}'\
                             .format(
                                 early_check,
                                 best_epoch,
                                 best['loss'],
                                 best['rating_acc'],
                                 best['rating_f1'],
                                 best['flag_acc'],
                                 best['flag_f1'],
                                 best['comb_acc'],
                                 best['comb_f1']
                                 ))
        
        # test best model on test data
        self.model.load_state_dict(torch.load(best_path))
        test_results = self.evaluate(self.test_data, val=False)
        log.info('Test Information: | Loss: {:.4f} | Rating Accuracy: {:.4f} | Rating F1: {:.4f}'\
                 ' | Flag Accuracy: {:.4f} | Flag F1: {:.4f} | Combined Accuracy: {:.4f}'\
                     ' | Best Combined F1: {:.4f}'.format(
                         test_results['loss'],
                         test_results['rating_acc'],
                         test_results['rating_f1'],
                         test_results['flag_acc'],
                         test_results['flag_f1'],
                         test_results['comb_acc'],
                         test_results['comb_f1']
                         ))
        
        return best_path